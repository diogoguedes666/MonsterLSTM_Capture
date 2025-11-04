#!/bin/bash
echo "🔍 Checking Repository Status for Colab..."
echo ""

echo "✅ Essential files that SHOULD be in repo:"
echo "-------------------------------------------"
[ -f "dist_model_recnet.py" ] && echo "✓ dist_model_recnet.py" || echo "✗ MISSING: dist_model_recnet.py"
[ -d "CoreAudioML" ] && echo "✓ CoreAudioML/ directory" || echo "✗ MISSING: CoreAudioML/"
[ -f "Configs/RNN3.json" ] && echo "✓ Configs/RNN3.json" || echo "✗ MISSING: Configs/RNN3.json"
[ -f "requirements.txt" ] && echo "✓ requirements.txt" || echo "✗ MISSING: requirements.txt"
[ -f "GuitarAmp_Training_Colab.ipynb" ] && echo "✓ GuitarAmp_Training_Colab.ipynb" || echo "✗ MISSING: GuitarAmp_Training_Colab.ipynb"
echo ""

echo "❌ Files that should NOT be committed (.gitignore):"
echo "-------------------------------------------"
[ -d "Results" ] && echo "⚠ Results/ folder (should be ignored)" || echo "✓ No Results/"
[ -d "runs2" ] && echo "⚠ runs2/ folder (should be ignored)" || echo "✓ No runs2/"
echo ""

echo "📊 Git Status Summary:"
echo "-------------------------------------------"
git status --short | head -10
echo ""
echo "For full status: git status"
