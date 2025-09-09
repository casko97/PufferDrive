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

To train with pufferdrive, we need to convert the `json` files to map binaries. To do this, run 
```
python pufferlib/ocean/drive/drive.py
```
with the path to your data folder. One example has been added for reference. 