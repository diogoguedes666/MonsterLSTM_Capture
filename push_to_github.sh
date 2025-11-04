#!/bin/bash
echo "🚀 Pushing to GitHub..."
echo "Repository: https://github.com/diogoguedes666/Automated-GuitarAmpModelling-3.git"
echo ""
echo "Commits to push:"
git log origin/main..main --oneline 2>/dev/null || echo "  (will push all commits)"
echo ""
echo "Attempting push..."
git push -u origin main
