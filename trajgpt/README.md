# TrajGPT


## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

There might be some missing libraries that needs to be updated, sklearn version needs to be updated also
```bash
git clone https://github.com/WahabF/trajgpt.git
cd trajgpt
conda env create -f environment.yml
conda activate minigpt4
```

Make sure to edit the paths in 
[trajgpt/train_configs/debug.yaml](trajgpt/train_configs/debug.yaml)
and
[trajgpt/train_configs/train.yaml](trajgpt/train_configs/train.yaml)

### Training

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/train.yaml
```
