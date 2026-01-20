# python src/train.py --config configs/knn/physical_small_k5_dist.yaml
# python src/train.py --config configs/knn/physical_small_k10_manhattan.yaml
# python src/train.py --config configs/knn/physical_small_k20_uniform.yaml

# Medium Experiments - Physical Data
python src/train.py --config configs/xgboost/physical_medium.yaml
python src/train.py --config configs/knn/physical_medium.yaml
python src/train.py --config configs/random_forest/physical_medium.yaml
# python src/train.py --config configs/mlp/physical_medium.yaml
# python src/train.py --config configs/tab_transformer/physical_medium.yaml
# python src/train.py --config configs/ft_transformer/physical_medium.yaml
# python src/train.py --config configs/attention_mlp/physical_medium.yaml

# Medium Experiments - Network Data
python src/train.py --config configs/xgboost/network_medium.yaml
python src/train.py --config configs/knn/network_medium.yaml
python src/train.py --config configs/random_forest/network_medium.yaml
# python src/train.py --config configs/mlp/network_medium.yaml
# python src/train.py --config configs/tab_transformer/network_medium.yaml
# python src/train.py --config configs/ft_transformer/network_medium.yaml
# python src/train.py --config configs/attention_mlp/network_medium.yaml