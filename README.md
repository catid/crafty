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

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -U -r requirements.txt

pip install ninja
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

pip install git+https://github.com/catid/crafter.git#egg=crafter

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
