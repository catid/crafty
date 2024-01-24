import argparse

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

import crafter
import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_noreward-random/0')
parser.add_argument('--steps', type=float, default=1e6)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

env = crafter.Env()
env = crafter.Recorder(
    env, args.outdir,
    save_stats=True,
    save_episode=False,
    save_video=False,
)
action_space = env.action_space

print(f"Created crafter env. Loading model...")

from net import DINOv2Backbone, NeckFormer, ImageDecoder

model_backbone = DINOv2Backbone()
model_backbone.eval()
model_backbone.to(device)

model_neck = NeckFormer(d_in=1024, height=16, width=16, d_out=128, d_hidden=128, segments=16, depth=1, expansion_factor=4, dropout=0.1)
model_neck.to(device)

model_decoder = ImageDecoder()
model_decoder.to(device)

def print_model_size(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_parameters:,} trainable parameters")

print(f"Loaded model.  Preparing...")

# Define L1 Loss and Optimizer
loss_function = torch.nn.L1Loss()
neck_optimizer = torch.optim.Adam(model_neck.parameters(), lr=0.001)
decoder_optimizer = torch.optim.Adam(model_decoder.parameters(), lr=0.001)

class ImageDataset(Dataset):
    def __init__(self):
        self.target_images = []
        self.embedding = []
        self.backbone_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
            lambda x: x.unsqueeze(0),  # Add a batch dimension
        ])
        self.recon_transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        ])

    def add_image(self, image):
        image = Image.fromarray(image)
        target_image = self.recon_transform(image)

        image = self.backbone_transform(image)
        image = image.to(device)

        with torch.no_grad():
            embedding = model_backbone(image)

        self.embedding.append(embedding)
        self.target_images.append(target_image)

    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, idx):
        return self.target_images[idx], self.embedding[idx]

dataset = ImageDataset()

# Training function
def train_model(dataset):
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
        for target_images, embeddings in train_loader:
            target_images, embeddings = target_images.to(device), embeddings.to(device)

            neck_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            outputs = model_neck(embeddings)
            recon = model_decoder(outputs)

            loss = loss_function(recon, target_images)

            loss.backward()
            neck_optimizer.step()
            decoder_optimizer.step()

        # Validation phase
        model_neck.eval()
        model_decoder.eval()
        total_val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for target_images, embeddings in val_loader:
                target_images, embeddings = target_images.to(device), embeddings.to(device)

                outputs = model_neck(embeddings)
                recon = model_decoder(outputs)

                val_loss = loss_function(recon, target_images)
                total_val_loss += val_loss.item()
                num_batches += 1

        avg_val_loss = total_val_loss / num_batches
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Check if validation loss has increased
        if avg_val_loss >= prev_val_loss:
            break  # Stop training if validation loss increases
        prev_val_loss = avg_val_loss

    #dataset.target_images.clear()
    #dataset.embedding.clear()
    torch.cuda.empty_cache()


print(f"Go!")

done = True
step = 0
bar = tqdm.tqdm(total=args.steps, smoothing=0)
while step < args.steps or not done:
    if done:
        done = False
        obs, info = env.reset()

    # Random action
    action = action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    dataset.add_image(obs)

    done = terminated or truncated

    if len(dataset) % 1000 == 0:
        train_model(dataset)

    step += 1
    bar.update(1)
