# Installation

## Recommended Environment

```bash
git clone https://github.com/Vision-CAIR/iMotion-LLM.git
cd iMotion-LLM

conda env create -f environment.yml
conda activate imotion-llm
```

If you prefer manual installation:

```bash
conda create -n imotion-llm python=3.9 -y
conda activate imotion-llm

pip install --upgrade pip setuptools wheel
pip install -r requirements/base.txt
pip install -r requirements/mtr.txt
```

## MTR CUDA Extensions

The `mtr/` package builds CUDA extensions and should be installed editable after the main environment is ready:

```bash
pip install -e ./mtr
```

If the build cannot find CUDA, set `CUDA_HOME` to your local CUDA toolkit path before running the command above.

## Optional Packages

- `flash-attn` can help on supported GPUs, but it is optional for the current public release.
- `waymo-open-dataset-tf-2-11-0` is only needed for the Waymo preprocessing path.
- If you are not using MTR, you can skip the MTR editable install step.
