#!/usr/bin/env python3
"""Execute a queue of training jobs sequentially."""

import configparser
import shutil
import subprocess
import sys
from pathlib import Path


def estimate_training_time(config_path, sps_estimate=23500):
    """Estimate training time from config.
    
    Args:
        config_path: Path to .ini config file
        sps_estimate: Estimated steps per second (default: 50k, adjust based on your hardware)
    
    Returns:
        Estimated time in seconds
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    total_timesteps = eval(config.get("train", "total_timesteps", fallback="5_000_000"))
    num_workers = int(config.get("vec", "num_workers", fallback="6"))
    
    # Effective SPS scales with workers
    effective_sps = sps_estimate * num_workers / 6
    return total_timesteps / effective_sps


def format_time(seconds):
    """Format seconds as human-readable time."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def run_training_queue(config_files, config_dest="config/ocean/drive.ini", sps_estimate=23500):
    """Run training jobs sequentially by copying configs and launching training.
    
    Args:
        config_files: List of .ini file paths to run
        config_dest: Destination path for the active config
        sps_estimate: Estimated steps per second for time estimation
    """
    config_dest = Path(config_dest)
    
    # Pre-compute all estimates
    estimates = []
    total_time = 0
    for config_path in config_files:
        config_path = Path(config_path)
        if config_path.exists():
            est_time = estimate_training_time(config_path, sps_estimate)
            estimates.append((config_path, est_time))
            total_time += est_time
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"Training Queue Summary: {len(estimates)} jobs")
    print(f"Total estimated time: {format_time(total_time)}")
    print(f"{'='*60}")
    for i, (config_path, est_time) in enumerate(estimates, 1):
        print(f"  {i}. {config_path.name}: {format_time(est_time)}")
    print(f"{'='*60}\n")
    
    for i, (config_path, est_time) in enumerate(estimates, 1):
        
        print(f"\n{'='*60}")
        print(f"Starting Job {i}/{len(estimates)}: {config_path.name}")
        print(f"Estimated time: {format_time(est_time)}")
        print(f"{'='*60}\n")
        
        # Copy config to active location
        shutil.copy2(config_path, config_dest)
        print(f"Copied {config_path} -> {config_dest}")
        
        # Launch training
        result = subprocess.run(["puffer", "train", "puffer_drive", "--wandb", "--wandb-project", "pufferdrive"])
        
        if result.returncode != 0:
            print(f"\nWarning: Job {i} exited with code {result.returncode}")
    
    print(f"\n{'='*60}")
    print(f"Completed all {len(estimates)} jobs")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python queue_training.py <config1.ini> <config2.ini> ...")
        sys.exit(1)
    
    run_training_queue(sys.argv[1:])
