# crafty

Leveraging foundation models to do well in the crafter gym.

## Setup

Install CUDA toolkit:

```bash
# Here's how to do it on Ubuntu:
sudo apt install nvidia-cuda-toolkit
```

Make sure you can use `nvcc`:

```bash
nvcc --version
```

```bash
git clone https://github.com/catid/crafty
cd crafty

conda create -n crafty python=3.10
conda activate crafty

pip install -U -r requirements.txt

# Update this from https://github.com/NVIDIA/DALI#installing-dali
pip install --upgrade nvidia-dali-cuda110 --extra-index-url https://developer.download.nvidia.com/compute/redist
```

## Train

```bash
python train_recon.py
```

Kill hanging processes:

```bash
pkill -f train_recon.py
```
