# PufferDrive


## Installation

Clone the repo
```bash
https://github.com/Emerge-Lab/PufferDrive.git
```

Make a venv
```
uv venv
```

Activate the venv
```
source .venv/bin/activate
```

Inside the venv, install the dependencies
```
uv pip install -e .
```

Compile the C code
```
python setup.py build_ext --inplace --force
```

To test your setup, you can run
```
puffer train puffer_drive
```

Alternative options for working with pufferdrive are found at https://puffer.ai/docs.html


## Quick start

Start a training run
```
puffer train puffer_drive
```

## Dataset

### Data preparation

To train with PufferDrive, you need to convert JSON files to map binaries. Run the following command with the path to your data folder:

```bash
python pufferlib/ocean/drive/drive.py
```

### Downloading Waymo Data

You can download the WOMD data from Hugging Face in two versions:

- **Mini Dataset**: [GPUDrive_mini](https://huggingface.co/datasets/EMERGE-lab/GPUDrive_mini) contains 1,000 training files and 300 test/validation files
- **Full Dataset**: [GPUDrive](https://huggingface.co/datasets/EMERGE-lab/GPUDrive) contains 100,000 unique scenes

**Note**: Replace 'GPUDrive_mini' with 'GPUDrive' in your download commands if you want to use the full dataset.

### Additional Data Sources

For more training data compatible with PufferDrive, see [ScenarioMax](https://github.com/valeoai/ScenarioMax). The GPUDrive data format is fully compatible with PufferDrive.
