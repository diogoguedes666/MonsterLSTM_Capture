# What to Commit to Your Repository for Colab

## ✅ **ESSENTIAL FILES** (Must be in repository):

### Core Training Code:
- `dist_model_recnet.py` - Main training script
- `CoreAudioML/` folder - All Python modules:
  - `CoreAudioML/__init__.py`
  - `CoreAudioML/dataset.py`
  - `CoreAudioML/miscfuncs.py`
  - `CoreAudioML/networks.py`
  - `CoreAudioML/training.py`
  - `CoreAudioML/test/` - Test files (optional but good practice)

### Configuration:
- `Configs/RNN3.json` - Training configuration
- `Configs/RNN1.json` - (if you use it)
- `Configs/RNN2.json` - (if you use it)

### Dependencies:
- `requirements.txt` - Python package dependencies

### Colab Setup:
- `GuitarAmp_Training_Colab.ipynb` - The Colab notebook
- `prepare_data_colab.py` - Helper script (optional)
- `COLAB_README.md` - Documentation (optional but helpful)

### Other Useful Files:
- `README.md` - Project documentation
- `proc_audio.py` - If you use it for inference
- `generate_test_signals.py` - If you use it

---

## ❌ **DO NOT COMMIT** (Add to .gitignore):

### Training Outputs (Generated Files):
- `Results/` folder - All trained models and checkpoints
- `runs2/` folder - TensorBoard logs
- `*.pt`, `*.pth` files - Model weights
- `checkpoint_epoch_*` files
- `training_stats.json` - Training statistics

### Audio Data (Too Large):
- `Data/train/*.wav` - Training audio files
- `Data/val/*.wav` - Validation audio files  
- `Data/test/*.wav` - Test audio files
- These can be uploaded separately to Colab or generated

### Temporary/Cache Files:
- `__pycache__/` folders
- `.DS_Store` files
- `*.log` files
- Test output folders

---

## 📋 **Quick Checklist:**

```bash
# 1. Create/update .gitignore (already done for you)
# 2. Add essential files:
git add CoreAudioML/
git add dist_model_recnet.py
git add Configs/RNN3.json
git add requirements.txt
git add GuitarAmp_Training_Colab.ipynb
git add prepare_data_colab.py
git add COLAB_README.md
git add README.md

# 3. Commit:
git commit -m "Add Colab training setup and essential files"

# 4. Push to GitHub:
git push origin main
```

---

## 🎯 **Next Steps:**

1. **Review your repository**: Make sure CoreAudioML/ folder is tracked
2. **Commit essential files**: Follow the checklist above
3. **Push to GitHub**: So Colab can clone it
4. **Upload notebook to Colab**: Open `GuitarAmp_Training_Colab.ipynb` in Colab
5. **Run training**: The notebook will clone your repo and run!

---

## 🔍 **Verify Before Pushing:**

Check what will be committed:
```bash
git status
git diff --cached  # See what's staged
```

If you see large files or Results/, they shouldn't be committed!

---

## 💡 **How Colab Will Use Your Repo:**

1. Colab clones your GitHub repository
2. Gets all the code (CoreAudioML/, dist_model_recnet.py, etc.)
3. Uploads audio data separately (or generates test data)
4. Runs training and saves Results/ locally (not in repo)
5. You download Results/ when done

**Audio files are NOT in the repo** - they're uploaded to Colab separately or generated on-the-fly!
