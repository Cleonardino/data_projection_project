#!/bin/bash
set -e

# Test all models on medium network data for 1 epoch with class weights

echo "Testing MLP..."
python src/train.py --config configs/mlp/network_medium.yaml --epochs 1 --balancing class_weights

echo "Testing AttentionMLP..."
python src/train.py --config configs/attention_mlp/network_medium.yaml --epochs 1 --balancing class_weights

echo "Testing TabTransformer..."
python src/train.py --config configs/tab_transformer/network_medium.yaml --epochs 1 --balancing class_weights

echo "Testing FT-Transformer..."
python src/train.py --config configs/ft_transformer/network_medium.yaml --epochs 1 --balancing class_weights

echo "✅ All tests completed!"
