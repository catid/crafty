import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dali_torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

import random
import numpy as np

# Sharding, batching external input dataset iterator
class ExternalInputIterator:
    def __init__(self, dataset, batch_size, shard_id, num_shards):
        self.dataset = dataset
        self.indices = []
        self.start_idx = -1
        self.end_idx = -1
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards

    def reset(self, seed):
        # This list of shuffled indices will be synchronized between all instances
        total_count = len(self.dataset)
        self.indices = list(range(total_count))
        rng = random.Random(seed)
        rng.shuffle(self.indices)

        # We select a subset of the indices to train on
        data_per_shard = (total_count + self.num_shards - 1) // self.num_shards
        self.start_idx = self.shard_id * data_per_shard
        if self.shard_id == self.num_shards - 1:
            self.end_idx = total_count
        else:
            self.end_idx = self.start_idx + data_per_shard

        #print(f"reset shard_id={self.shard_id} total_count={total_count} data_per_shard={data_per_shard} start_idx={self.start_idx} end_idx={self.end_idx}")

    def __iter__(self):
        self.i = self.start_idx
        return self

    def __next__(self):
        if self.i >= self.end_idx:
            raise StopIteration

        batch_count = min(self.batch_size, self.end_idx - self.i)
        batch = [self.dataset[self.i + j] for j in range(batch_count)]
        self.i += batch_count

        return batch

@pipeline_def(batch_size=64, num_threads=8, device_id=0)
def in_memory_pipeline(image_iterator, embed_iterator, crop_w=224, crop_h=224, fp16=False):
    dtype = types.FLOAT16 if fp16 else types.FLOAT

    images = fn.external_source(
        source=image_iterator,
        device="gpu",
        name="Reader")

    # Original transformations
    images_resized = fn.resize(images, resize_x=crop_w, resize_y=crop_h, interp_type=types.INTERP_TRIANGULAR)
    images_normalized = fn.crop_mirror_normalize(images_resized,
                                      dtype=dtype,
                                      output_layout="CHW",
                                      crop=(crop_h, crop_w),
                                      mean=[123.675, 116.28, 103.53],
                                      std=[58.395, 57.12, 57.375])

    # Alternative transformation: Scale to -1 to 1 range without cropping
    images_scaled = fn.crop_mirror_normalize(images_resized, 
                                 dtype=dtype,
                                 mean=[127.5, 127.5, 127.5],
                                 std=[127.5, 127.5, 127.5])

    cached_embeddings = fn.external_source(source=embed_iterator, device='cpu')

    return images_normalized, images_scaled, cached_embeddings

class CustomInMemoryDALILoader:
    def __init__(self,
                 image_iterator, embed_iterator,
                 batch_size=64, num_threads=8, device_id=0,
                 crop_w=224, crop_h=224, fp16=False):

        self.pipeline = in_memory_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            image_iterator=image_iterator,
            embed_iterator=embed_iterator,
            crop_w=crop_w,
            crop_h=crop_h,
            fp16=fp16)

        self.pipeline.build()

        self.loader = dali_torch.DALIGenericIterator(
            self.pipeline,
            ["normalized", "scaled", "embeddings"],
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL)

    def __iter__(self):
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)
