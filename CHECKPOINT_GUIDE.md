# Checkpoint System Guide

## Overview

The experiment now automatically saves a checkpoint after solving the Lyapunov equation (the most time-consuming step). If the experiment crashes or is interrupted during training, you can resume without redoing the expensive computation.

## What Gets Saved

The checkpoint includes:
- Kernel matrix `K`
- Coefficient matrix `C` from kernel ridge regression
- Lyapunov solution matrix `B`
- Transformation matrix `T_H`
- Training data `X_train`
- Target representations `F_target`

**Location**: `results/checkpoints/generator_checkpoint.npz`

## How to Use

### First Run (Normal)
```bash
python scripts/run_experiment.py
```

This will:
1. Load CIFAR-100 data
2. Extract target representations from ResNet-18
3. Generate optimal augmentations (solve Lyapunov equation)
4. **Save checkpoint automatically**
5. Train with Barlow Twins loss
6. Save final results

### Resume from Checkpoint
```bash
python scripts/run_experiment.py --resume
```

This will:
1. Load CIFAR-100 data (quick)
2. Extract target representations (quick)
3. **Load checkpoint** (skips expensive Lyapunov equation solving!)
4. Train with Barlow Twins loss
5. Save final results

## When is the Checkpoint Saved?

The checkpoint is saved **immediately after** the Lyapunov equation is solved, right before training begins.

You'll see this message:
```
Saving checkpoint to: results/checkpoints/generator_checkpoint.npz
Checkpoint saved successfully!

================================================================================
STEP 4: TRAINING WITH BARLOW TWINS LOSS
================================================================================
```

## Time Savings

Approximate times on typical hardware:
- **Steps 1-3** (data loading + Lyapunov solving): ~2-5 minutes
- **Step 4** (Training 3000 epochs): ~15-20 minutes

**With resume**: Skip 2-5 minutes and jump straight to training!

## Example Usage

### Experiment Interrupted During Training

```bash
# Original run (interrupted at epoch 1500/3000)
$ python scripts/run_experiment.py
...
Training:  50%|████████████████████  | 1500/3000 [09:34<08:24,  2.97it/s]
^C  # User interrupted or crash

# Resume (skip to training immediately)
$ python scripts/run_experiment.py --resume

Found checkpoint at: results/checkpoints/generator_checkpoint.npz
Loading from checkpoint...
Checkpoint loaded successfully!
Skipping Lyapunov equation solving...

Augmentation Distribution Info:
  Min eigenvalue: 0.000000e+00
  Max eigenvalue: 7.404423e+00
  Condition number: 7.404423e+10

================================================================================
STEP 4: TRAINING WITH BARLOW TWINS LOSS
================================================================================
Training for 3000 epochs...
```

### Force Fresh Start

If you want to start from scratch even if a checkpoint exists:

```bash
# Don't use --resume flag
python scripts/run_experiment.py
```

Or delete the checkpoint:
```bash
rm results/checkpoints/generator_checkpoint.npz
python scripts/run_experiment.py
```

## Notes

- The checkpoint is **automatically** saved, you don't need to do anything
- Use `--resume` only if you want to skip the Lyapunov equation solving
- The checkpoint is specific to the configuration file used
- If you change kernel parameters or data size, you should delete the old checkpoint

## Troubleshooting

### "No checkpoint found" when using --resume
Make sure you've run the experiment at least once to completion of Step 3. The checkpoint is only saved after the Lyapunov equation is solved.

### Checkpoint seems corrupted
Delete it and start fresh:
```bash
rm results/checkpoints/generator_checkpoint.npz
python scripts/run_experiment.py
```

### Want different experiment config
Either:
1. Delete the checkpoint and run with new config
2. Use a different output directory in your config file
