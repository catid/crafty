import argparse
import multiprocessing

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

import crafter
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_noreward-random/0')
parser.add_argument('--steps', type=int, default=1e6)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

from net import DINOv2Backbone, NeckFormer, ImageDecoder

class ImageDataset(Dataset):
    def __init__(self, model_backbone):
        self.model_backbone = model_backbone

        self.images = []
        self.backbone_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])
        self.recon_transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        ])

    def add_image(self, image):
        image = Image.fromarray(image)
        self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        target_image = self.recon_transform(image)
        input_image = self.backbone_transform(image)

        return input_image, target_image

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

        queue.put(obs)  # Add the observation to the queue

def train_model(queue, total_steps):
    model_backbone = DINOv2Backbone()
    model_backbone.eval()
    model_backbone.to(device)

    model_neck = NeckFormer(d_in=1024, height=16, width=16, d_out=256, d_hidden=256, segments=16, depth=2, expansion_factor=4, dropout=0.1)
    model_neck.to(device)

    model_decoder = ImageDecoder(d_in=256, d_chan=16, d_conv=4)
    model_decoder.to(device)

    def print_model_size(model, model_name):
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{model_name} has {num_parameters:,} trainable parameters")

    print_model_size(model_neck, "model_neck")
    print_model_size(model_decoder, "model_decoder")

    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) > 1:
        model_backbone = nn.DataParallel(model_backbone, device_ids=device_ids)
        model_neck = nn.DataParallel(model_neck, device_ids=device_ids)
        model_decoder = nn.DataParallel(model_decoder, device_ids=device_ids)

    # Define L1 Loss and Optimizer
    loss_function = torch.nn.L1Loss()
    neck_optimizer = torch.optim.Adam(model_neck.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.Adam(model_decoder.parameters(), lr=0.001)

    dataset = ImageDataset(model_backbone)

    print(f"Loaded model.  Awaiting training data...")

    for step in tqdm(range(total_steps), desc="Training Progress"):
        obs = queue.get()

        dataset.add_image(obs)

        if len(dataset) % 1000 != 0:
            continue

        # Split dataset into training and validation
        train_size = int(0.8 * len(dataset))  # 80% of data for training
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        prev_val_loss = float('inf')
        while True:
            # Training phase
            model_neck.train()
            model_decoder.train()
            for input_batch, target_batch in train_loader:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                neck_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                with torch.no_grad():
                    embeddings = model_backbone(input_batch)

                outputs = model_neck(embeddings)
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
                for input_batch, target_batch in val_loader:
                    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                    with torch.no_grad():
                        embeddings = model_backbone(input_batch)
                    outputs = model_neck(embeddings)
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


def main():
    N = 2  # Number of processes
    queue = multiprocessing.Queue()
    total_steps = int(1e6)
    outdir_base = 'logdir/crafter_noreward-random/'

    processes = []

    # Dynamically creating N processes
    for i in range(N):
        outdir = f"{outdir_base}{i}"
        steps_per_process = total_steps // N
        process = multiprocessing.Process(target=run_environment, args=(queue, steps_per_process, outdir))
        processes.append(process)
        process.start()

    # Run the training in the main process
    train_model(queue, total_steps)

    try:
        # Wait for all environment processes to finish
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
            process.join()

if __name__ == "__main__":
    main()
