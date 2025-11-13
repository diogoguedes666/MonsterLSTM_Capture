# Fine-Tuning Guide for Guitar Amp Modelling

## Overview

Fine-tuning allows you to continue training a pre-trained model on new data or different datasets. This guide explains how to fine-tune your models, what datasets you can use, and important considerations.

## How to Fine-Tune

### Method 1: Fine-Tune on Same Dataset (Continue Training)

If you want to continue training an existing model on the same dataset:

```bash
python dist_model_recnet.py \
    --load_config RNN3 \
    --data_location ./Data \
    --file_name dls2 \
    --load_model True \
    --learn_rate 0.0001  # Use lower learning rate for fine-tuning
```

The model will automatically:
- Load the existing model from `Results/{device}-{config}/model.json`
- Resume training from the last checkpoint
- Continue with the same train/val/test splits

### Method 2: Fine-Tune on Different Dataset

To fine-tune a pre-trained model on a different dataset:

```bash
python dist_model_recnet.py \
    --load_config RNN3 \
    --data_location ./pedal_training_data \  # New dataset location
    --file_name pedal \                      # New file name
    --device pedal \                         # New device name (optional)
    --load_model True \
    --learn_rate 0.00005  # Lower learning rate for fine-tuning
```

**Important**: The new dataset must have the same structure:
```
new_dataset/
├── train/
│   ├── {file_name}-input.wav
│   └── {file_name}-target.wav
├── val/
│   ├── {file_name}-input.wav
│   └── {file_name}-target.wav
└── test/
    ├── {file_name}-input.wav
    └── {file_name}-target.wav
```

### Method 3: Fine-Tune with Specific Checkpoint

To load a specific checkpoint instead of the latest model:

1. Copy the checkpoint file to the model directory
2. Modify the code to load from a specific checkpoint, or
3. Use the existing model.json which points to the best model

## Using Different Datasets

### ✅ You CAN Use Different Datasets

You can fine-tune on:
- **Different audio content**: Different guitar tones, pedals, or amplifiers
- **Different audio characteristics**: Different frequency responses, distortion levels
- **Larger or smaller datasets**: More or less training data

### ⚠️ Important Considerations

1. **Sample Rate Compatibility**
   - All datasets must have the same sample rate
   - The model expects consistent sample rates across train/val/test

2. **Audio Characteristics**
   - Fine-tuning works best when the new dataset is similar to the original
   - Very different audio characteristics may require lower learning rates
   - Consider using a learning rate 5-10x lower than initial training

3. **Model Architecture Compatibility**
   - The model architecture (hidden_size, unit_type, etc.) must match
   - Check your config file matches the pre-trained model

4. **Potential Downsides**
   - **Catastrophic Forgetting**: Model may forget original dataset characteristics
   - **Overfitting**: If new dataset is small, model may overfit
   - **Domain Shift**: Very different audio may not transfer well
   - **Loss Function Mismatch**: If new data has different characteristics, loss weights may need adjustment

### Best Practices for Fine-Tuning

1. **Use Lower Learning Rate**: Start with 0.00005-0.0001 (vs 0.0001-0.001 for initial training)
2. **Monitor Validation Loss**: Watch for overfitting on new data
3. **Consider Mixed Training**: Combine old and new datasets if possible
4. **Adjust Loss Weights**: You may need to adjust loss function weights in your config
5. **Save Separate Models**: Keep original model separate from fine-tuned version

## Dataset Files: What You Should NOT Change

### ❌ DO NOT CHANGE: Test Set

**The test set should remain UNCHANGED** for the following reasons:

1. **Fair Evaluation**: Test set provides unbiased evaluation of model performance
2. **Comparability**: Allows comparison between different training runs
3. **Overfitting Detection**: Helps detect if model is overfitting to validation set
4. **Reproducibility**: Enables reproducible evaluation metrics

**Best Practice**: 
- Keep the same test set across all experiments
- Only use test set for final evaluation, never for training decisions
- If you need to evaluate on new data, create a separate evaluation set

### ✅ You CAN Change: Train and Validation Sets

- **Train Set**: Can be modified, augmented, or replaced
- **Validation Set**: Can be modified, but keep it representative of your target domain

### Recommended Dataset Strategy

```
Original Dataset (for comparison):
├── test/          # NEVER CHANGE - used for final evaluation
├── train/         # Can modify for fine-tuning
└── val/           # Can modify, but keep representative

New Dataset (for fine-tuning):
├── test/          # Use original test set OR create new one
├── train/         # Your new training data
└── val/           # Your new validation data
```

## Example Fine-Tuning Workflow

### Step 1: Prepare Your New Dataset

```bash
# Ensure your new dataset follows the structure:
new_dataset/
├── train/
│   ├── mypedal-input.wav
│   └── mypedal-target.wav
├── val/
│   ├── mypedal-input.wav
│   └── mypedal-target.wav
└── test/
    ├── mypedal-input.wav
    └── mypedal-target.wav
```

### Step 2: Create or Modify Config File

Create `Configs/RNN3_finetune.json`:

```json
{
  "hidden_size": 64,
  "unit_type": "LSTM",
  "loss_fcns": {"ESR": 0.60, "DC": 0.10, "HFHinge": 0.30},
  "pre_filt": "None",
  "device": "mypedal",
  "file_name": "mypedal",
  "learn_rate": 0.00005,
  "hf_hinge_fmin": 8000,
  "hf_hinge_strength": 1.0
}
```

### Step 3: Run Fine-Tuning

```bash
python dist_model_recnet.py \
    --load_config RNN3_finetune \
    --data_location ./new_dataset \
    --file_name mypedal \
    --load_model True \
    --learn_rate 0.00005 \
    --epochs 500
```

### Step 4: Monitor Training

- Watch validation loss - should decrease smoothly
- If validation loss increases, reduce learning rate
- Save checkpoints regularly
- Compare test results with original model

## Troubleshooting

### Problem: Model Performance Degrades After Fine-Tuning

**Solutions**:
- Lower learning rate further (try 0.00001)
- Reduce number of epochs
- Use learning rate scheduling
- Consider training on mixed dataset (old + new)

### Problem: Model Overfits Quickly

**Solutions**:
- Increase weight decay
- Use more data augmentation
- Reduce model capacity
- Early stopping with patience

### Problem: Model Doesn't Learn New Dataset

**Solutions**:
- Check dataset format and sample rate
- Verify model architecture matches
- Try higher learning rate initially, then reduce
- Check if dataset is too different from original

## Summary

- ✅ **Fine-tuning is supported** - use `--load_model True`
- ✅ **You can use different datasets** - specify with `--data_location` and `--file_name`
- ✅ **You can modify train/val sets** - but keep test set unchanged
- ❌ **DO NOT change test set** - it's for fair evaluation
- ⚠️ **Use lower learning rates** - typically 0.00005-0.0001 for fine-tuning
- ⚠️ **Monitor for overfitting** - watch validation loss carefully
- ⚠️ **Consider dataset similarity** - very different datasets may not transfer well

