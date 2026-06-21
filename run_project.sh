echo "Feature-Extraktion: run extract_features...."
python scripts/extract_features.py

echo "Training: Early Fusion: run train_early...."
python scripts/train_early.py

echo "Training: Cross-Attention: run train_cross...."
python scripts/train_cross.py