# Berzelius Setup

Berzelius lathund:
* Monitor job status: `squeue -u $USER`
* Submit job: `sbatch myjob.sh`
* Interactive session `interactive -A proj-id -t 1:00:00 --gpus=1 --constraint thin`
* Launch bash: `srun --pty bash`
* Get into the node from the side: `srun --jobid=id --pty bash`

## First time environment creation

1. Load mamba: `module load Miniforge3/24.7.1-2-hpc1-bdist`
2. Crete env: `mamba create -n puffer-nov.venv python=3.11` `mamba create -n puffer-nov-cu121.venv python=3.11`
3. Activate env: `mamba activate puffer-nov.venv` 
`mamba activate puffer-nov-cu121.venv`
4. Setup torch and CUDA: `CONDA_OVERRIDE_CUDA=13.0 mamba install pytorch==2.14.1=cuda*` 
`mamba install --override-channels -c pytorch -c nvidia -c conda-forge pytorch==2.5.1 pytorch-cuda=12.1`
5. Load gcc: `module load buildenv-gcccuda/12.1.1-gcc12.3.0`
6. Install req's: `uv pip install -e .` `python -m pip install -e . --no-build-isolation`
7. Compile C code: `python setup.py build_ext --inplace --force` 

## Using an existing environment 

1. Load mamba: `module load Miniforge3/24.7.1-2-hpc1-bdist`
2. Load gcc: `module load buildenv-gcccuda/12.1.1-gcc12.3.0`
3. Activate env: `mamba activate puffer-nov.venv`

