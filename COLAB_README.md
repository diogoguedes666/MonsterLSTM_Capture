# Running Guitar Amp Modelling Training on Google Colab

This guide shows you how to run your PyTorch guitar amplifier modelling training on Google Colab with free GPU acceleration.

## Quick Start

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Upload the notebook**: Upload `GuitarAmp_Training_Colab.ipynb` from this repository

3. **Enable GPU**: Click `Runtime` → `Change runtime type` → Select `GPU` under Hardware accelerator

4. **Run all cells**: Click `Runtime` → `Run all` or run cells one by one

## What the Notebook Does

- ✅ Installs PyTorch with CUDA support
- ✅ Clones your repository
- ✅ Generates test audio data (or you can upload your own)
- ✅ Creates necessary configuration files
- ✅ Runs the training with optimized settings for Colab
- ✅ Saves results for download

## Using Your Own Data

Instead of synthetic test data, upload real guitar recordings:

1. Record guitar → amp chains as stereo WAV files (44.1kHz)
2. Left channel: clean guitar signal
3. Right channel: amp output (or separate files)
4. Upload to Colab using the file upload cells

## Configuration Options

The notebook uses `RNN3.json` config by default. You can modify:

- `hidden_size`: Model complexity (32, 64, 128)
- `batch_size`: Memory usage (512, 1024, 2048)
- `epochs`: Training duration (50-500)
- `learn_rate`: Learning rate (0.001-0.01)

## Expected Training Times

With Colab's free T4 GPU:
- Small model (32 hidden): ~5-15 minutes per 100 epochs
- Medium model (64 hidden): ~10-30 minutes per 100 epochs
- Large model (128 hidden): ~20-60 minutes per 100 epochs

## Downloading Results

After training completes:
1. Zip the Results folder: `!zip -r results.zip Results/`
2. Download: `files.download('results.zip')`

## Troubleshooting

**"CUDA out of memory"**: Reduce batch_size in config
**"No GPU detected"**: Make sure GPU runtime is enabled
**"Import errors"**: Check all files are uploaded correctly
**Slow training**: Colab's free tier has limited GPU time (12+ hours max)

## Colab Pro Tips

- Use Colab Pro for longer training sessions
- Save checkpoints regularly to avoid losing progress
- Monitor GPU usage with `!nvidia-smi` in a separate cell
- Use TensorBoard for real-time training visualization

## Next Steps

1. Download trained models
2. Test with `proc_audio.py` on new guitar recordings
3. Experiment with different architectures
4. Fine-tune hyperparameters for better sound quality
