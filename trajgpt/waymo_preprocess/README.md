## Dataset and Environment
### 1. Download
- Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/) v1.1. Utilize data from ```scenario/training_20s``` or ```scenario/training``` for training, and data from ```scenario/validation``` and ```scenario/validation_interactive``` for testing.
- navigate to the directory:
```
cd waymo_preprocess
```

### 2. Environment Setup
- Create a conda environment:
```
conda create -n gameformer python=3.8
```
- Activate the conda environment:
```
conda activate gameformer
```
- Install the required packages:
```
pip install -r requirements.txt
```