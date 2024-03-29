import crafter
from PIL import Image
from tqdm import tqdm
import argparse
import time


################################################################################
# Torch

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os, random

import deepspeed
from deepspeed import comm
from deepspeed import log_dist
from deepspeed.runtime.config import DeepSpeedConfig

# Prettify printing tensors when debugging
import lovely_tensors as lt
lt.monkey_patch()

# Deepspeed logging functions
def log_0(msg):
    log_dist(msg, ranks=[0])
def log_all(msg):
    log_dist(msg, ranks=[-1])

def is_main_process():
    return comm.get_rank() == 0

# Enable cuDNN benchmarking to improve online performance
torch.backends.cudnn.benchmark = True

# Disable profiling to speed up training
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

def get_true_random_32bit_positive_integer():
    random_bytes = bytearray(os.urandom(4))
    random_bytes[0] &= 0x7F # Clear high bit
    random_int = int.from_bytes(bytes(random_bytes), byteorder='big')
    return random_int

def synchronize_seed(args, rank, shard_id):
    if args.seed < 0:
        seed = get_true_random_32bit_positive_integer()
    else:
        seed = args.seed

    if shard_id == 0:
        seed_tensor = torch.tensor(seed, dtype=torch.long)  # A tensor with the value to be sent
    else:
        seed_tensor = torch.zeros(1, dtype=torch.long)  # A tensor to receive the value

    seed_tensor = seed_tensor.cuda(rank)

    comm.broadcast(tensor=seed_tensor, src=0)

    seed = int(seed_tensor.item()) + shard_id
    args.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_all(f"Using seed: {seed} for shard_id={shard_id}")
    return seed


################################################################################
# Model Training

from net import DINOv2Backbone, NeckFormer, ImageDecoder
from dataloader import CustomInMemoryDALILoader, ExternalInputIterator

def make_optimizer(model, args):
    model_parameters = list(model.parameters())
    optimizer_params = [p for p in model_parameters if not hasattr(p, "_optim")]
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def print_model_size(model, model_name):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_0(f"{model_name} has {num_parameters:,} trainable parameters")

def ref_forward_and_loss(embeddings, target_image, model_neck, model_decoder, criterion):
    # DeepSpeed: forward + backward + optimize
    outputs = model_neck(embeddings)
    pred_image = model_decoder(outputs)
    return criterion(pred_image, target_image), pred_image

def train_model(args):
    t0 = time.time()

    deepspeed.init_distributed(
        dist_backend="nccl",
        verbose="false"
    )

    model_backbone = DINOv2Backbone()
    model_backbone.eval()

    model_neck = NeckFormer(d_in=1024, height=16, width=16, d_out=256, d_hidden=256, segments=16, depth=2, expansion_factor=4, dropout=0.1)
    neck_optimizer = make_optimizer(model_neck, args)
    neck_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(neck_optimizer, T_max=args.max_epochs)

    model_decoder = ImageDecoder(d_in=256, d_chan=16, d_conv=4)
    decoder_optimizer = make_optimizer(model_decoder, args)
    decoder_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=args.max_epochs)

    print_model_size(model_neck, "model_neck")
    print_model_size(model_decoder, "model_decoder")

    # DeepSpeed engine
    neck_engine, neck_engine_optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model_neck,
        optimizer=neck_optimizer,
        lr_scheduler=neck_lr_scheduler,
        #config_params=args.deepspeed_config,  <- This should be in the args
        model_parameters=model_neck.parameters())

    decoder_engine, decoder_engine_optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model_decoder,
        optimizer=decoder_optimizer,
        lr_scheduler=decoder_lr_scheduler,
        #config_params=args.deepspeed_config,  <- This should be in the args
        model_parameters=model_decoder.parameters())

    log_0(f"Arguments: {args}")

    comm.barrier()

    fp16 = decoder_engine.fp16_enabled()
    log_0(f'decoder_engine.fp16_enabled={fp16}')

    if fp16:
        image_dtype = torch.float16
    else:
        image_dtype = torch.float32

    rank = neck_engine.local_rank
    assert decoder_engine.local_rank == rank, "Unexpected mismatch"
    shard_id = neck_engine.global_rank
    assert decoder_engine.global_rank == shard_id, "Unexpected mismatch"
    num_gpus = neck_engine.world_size
    train_batch_size = neck_engine.train_batch_size()
    data_loader_batch_size = neck_engine.train_micro_batch_size_per_gpu()
    steps_per_print = neck_engine.steps_per_print()

    num_loader_threads = os.cpu_count()//2
    crop_w = 224
    crop_h = 224
    val_split_ratio = 0.2
    min_train_size = 1024
    min_val_size = 128

    seed = synchronize_seed(args, rank, shard_id)

    log_all(f"rank = {rank}, num_shards = {num_gpus}, shard_id={shard_id}, train_batch_size = {train_batch_size}, data_loader_batch_size = {data_loader_batch_size}, steps_per_print = {steps_per_print}, seed={seed}")

    # Define L1 Loss and Optimizer
    criterion = torch.nn.L1Loss()
    criterion.cuda(rank)

    forward_and_loss = torch.compile(ref_forward_and_loss, dynamic=True, fullgraph=True)

    full_dataset = []
    full_embeddings = []

    def make_random_image():
        # Create an array of shape (64, 64, 3) with random values
        array = np.random.rand(64, 64, 3) * 255
        array = array.astype(np.uint8)
        return array

    # Make some mock data
    for _ in range(2000):
        full_dataset.append(make_random_image())
        full_embeddings.append(make_random_image())

    image_iterator = ExternalInputIterator(full_dataset, data_loader_batch_size, shard_id, num_gpus)
    embed_iterator = ExternalInputIterator(full_embeddings, data_loader_batch_size, shard_id, num_gpus)

    image_iterator.reset(seed=0)
    embed_iterator.reset(seed=0)

    train_loader = CustomInMemoryDALILoader(
        image_iterator,
        embed_iterator,
        batch_size=data_loader_batch_size,
        num_threads=num_loader_threads,
        device_id=rank,
        crop_w=crop_w,
        crop_h=crop_h,
        fp16=fp16)

    # Now you can use train_loader and val_loader in your training and validation loops
    # Example:
    for epoch in range(4):
        print(f"RESET epoch={epoch}")
        image_iterator.reset(seed=epoch)
        embed_iterator.reset(seed=epoch)

        # Training loop
        for batch in train_loader:
            batch = batch[0]
            normalized, scaled, embeddings = batch["normalized"], batch["scaled"], batch["embeddings"]
            print(f"Batch epoch={epoch}")
            #print(f"normalized = {normalized}")
            #print(f"scaled = {scaled}")
            #print(f"embeddings = {embeddings}")

    return


    dataset = ImageDataset(model_backbone)

    embeddings_cache = {}

    log_0(f"Loaded model.  Awaiting training data...")

    for step in tqdm(range(total_steps), desc="Training Progress"):
        obs = queue.get()

        dataset.add_image(obs)

        if len(dataset) % training_interval != 0:
            continue

        # Split dataset into training and validation
        train_size = int(0.8 * len(dataset))  # 80% of data for training
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        def get_cached_batch_embeddings(indices_batch, cache_batch_size=16):
            # Convert indices_batch to a list for indexing
            indices_batch_list = indices_batch.tolist()

            # Initialize the embeddings list
            embeddings_list = [None] * len(indices_batch)

            # Identify uncached indices and prepare smaller batches of them
            uncached_indices = [index for index in indices_batch_list if index not in embeddings_cache]
            sub_batches = [uncached_indices[i:i+cache_batch_size] for i in range(0, len(uncached_indices), cache_batch_size)]

            # Process each sub-batch and update the cache
            for sub_batch in sub_batches:
                if len(sub_batch) <= 0:
                    continue

                # Prepare the sub-batch for processing
                sub_batch_input = torch.stack([input_batch[indices_batch_list.index(index)] for index in sub_batch])

                # Process the sub-batch
                with torch.no_grad():
                    sub_batch_embeddings = model_backbone(sub_batch_input)

                # Update cache and embeddings_list
                for i, index in enumerate(sub_batch):
                    embeddings = sub_batch_embeddings[i]
                    embeddings_cache[index] = embeddings
                    embeddings_list[indices_batch_list.index(index)] = embeddings

            # Fill in the cached embeddings for the rest of the batch
            for i, index in enumerate(indices_batch_list):
                if embeddings_list[i] is None:
                    embeddings_list[i] = embeddings_cache[index]

            return embeddings_list

        prev_val_loss = float('inf')
        while True:
            # Training phase
            model_neck.train()
            model_decoder.train()
            for input_batch, target_batch, indices_batch in train_loader:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                batch_embeddings = torch.stack(get_cached_batch_embeddings(indices_batch))
                batch_embeddings = batch_embeddings.to(device)

                neck_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                outputs = model_neck(batch_embeddings)
                recon = model_decoder(outputs)
                loss = loss_function(recon, target_batch)

                loss.backward()
                neck_optimizer.step()
                decoder_optimizer.step()

            # Validation phase
            model_neck.eval()
            model_decoder.eval()
            total_val_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for input_batch, target_batch, indices_batch in val_loader:
                    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                    batch_embeddings = torch.stack(get_cached_batch_embeddings(indices_batch))
                    batch_embeddings = batch_embeddings.to(device)

                    outputs = model_neck(batch_embeddings)
                    recon = model_decoder(outputs)

                    val_loss = loss_function(recon, target_batch)
                    total_val_loss += val_loss.item()
                    num_batches += 1

            avg_val_loss = total_val_loss / num_batches
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Check if validation loss has increased
            if avg_val_loss > prev_val_loss:
                break  # Stop training if validation loss increases
            prev_val_loss = avg_val_loss

        #dataset.target_images.clear()
        #dataset.embedding.clear()
        torch.cuda.empty_cache()


################################################################################
# Crafter Environment

def run_environment(queue, steps, outdir):
    env = crafter.Env()
    env = crafter.Recorder(env, outdir, save_stats=True, save_episode=False, save_video=False)
    action_space = env.action_space

    done = True
    for step in range(steps):
        if done:
            done = False
            obs, info = env.reset()

        action = action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        image = Image.fromarray(obs)

        queue.put(image)  # Add the observation to the queue


################################################################################
# Entrypoint

def main(args):
    train_model(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=1e6)
    parser.add_argument('--interval', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--max-epochs', type=int, default=128)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args)
