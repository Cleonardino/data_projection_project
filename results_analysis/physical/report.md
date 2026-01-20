# Analysis Report: Physical Dataset

Generated: 2026-01-20 22:16:05

---

## 📊 Model Leaderboard

Ranked by F1 (macro) score on test set.

| Rank | Experiment Name | Accuracy | F1 (macro) | Balanced Acc | MCC | Time (s) |
|------|---|---|---|---|---|---|
| 1 | `2026-01-20_20-35-18_physical_medium_xgboost` | 0.9872 | 0.8134 | 0.8075 | 0.9641 | 21.1 |
| 2 | `2026-01-20_21-54-31_physical_medium_tab_transformer` | 0.9780 | 0.8129 | 0.8270 | 0.9419 | 249.2 |
| 3 | `2026-01-20_20-26-50_physical_medium_knn` | 0.9793 | 0.8095 | 0.8249 | 0.9450 | 0.0 |
| 4 | `2026-01-20_20-35-43_physical_medium_knn` | 0.9793 | 0.8095 | 0.8249 | 0.9450 | 0.0 |
| 5 | `2026-01-20_20-18-39_physical_small_k5_dist_knn` | 0.9793 | 0.8095 | 0.8249 | 0.9450 | 0.0 |
| 6 | `2026-01-14_23-01-46_physical_small_knn` | 0.9774 | 0.8007 | 0.8235 | 0.9402 | 0.0 |
| 7 | `2026-01-14_23-38-28_physical_small_tab_transformer` | 0.9683 | 0.8001 | 0.8250 | 0.9188 | 40.7 |
| 8 | `2026-01-20_20-26-58_physical_medium_random_forest` | 0.9695 | 0.7936 | 0.8150 | 0.9198 | 0.6 |
| 9 | `2026-01-20_20-35-51_physical_medium_random_forest` | 0.9695 | 0.7936 | 0.8150 | 0.9198 | 0.7 |
| 10 | `2026-01-20_22-00-54_physical_medium_attention_mlp` | 0.9365 | 0.7612 | 0.8163 | 0.8510 | 108.1 |
| 11 | `2026-01-20_20-18-50_physical_small_k10_manhattan_knn` | 0.9390 | 0.7531 | 0.8164 | 0.8564 | 0.0 |
| 12 | `2026-01-14_21-23-07_physical_small_attention_mlp` | 0.9048 | 0.7259 | 0.8106 | 0.7946 | 35.2 |
| 13 | `2026-01-20_21-58-46_physical_medium_ft_transformer` | 0.8847 | 0.6974 | 0.7991 | 0.7601 | 121.0 |
| 14 | `2026-01-20_20-19-22_physical_small_k20_uniform_knn` | 0.8908 | 0.6969 | 0.8062 | 0.7722 | 0.0 |
| 15 | `2026-01-14_23-14-32_physical_small_mlp` | 0.8786 | 0.6889 | 0.7988 | 0.7515 | 22.7 |
| 16 | `2026-01-20_21-53-48_physical_medium_mlp` | 0.8700 | 0.6843 | 0.7946 | 0.7399 | 38.0 |
| 17 | `2026-01-14_22-00-45_physical_small_ft_transformer` | 0.8401 | 0.6593 | 0.7970 | 0.7004 | 56.1 |
| 18 | `2026-01-14_23-39-57_physical_small_xgboost` | 0.8676 | 0.6412 | 0.7906 | 0.7287 | 0.6 |
| 19 | `2026-01-14_23-16-06_physical_small_random_forest` | 0.6101 | 0.4687 | 0.7242 | 0.4779 | 0.2 |

---

## 🎯 Hard Classes Analysis

Classes with highest error rates across all models.

| Class | Total | Errors | Error Rate |
|-------|-------|--------|------------|
| scan | 38 | 38 | 100.00% |
| normal | 24681 | 2476 | 10.03% |
| nomal | 703 | 15 | 2.13% |
| MITM | 2869 | 61 | 2.13% |
| DoS | 893 | 7 | 0.78% |
| physical fault | 1957 | 9 | 0.46% |

---

## 🔍 Hardest Samples

Samples misclassified by most models (top 20).

| Sample | True Label | Times Wrong | Common Prediction |
|--------|------------|-------------|-------------------|
| 768 | normal | 19 | physical fault |
| 927 | normal | 19 | MITM |
| 848 | normal | 19 | physical fault |
| 1257 | normal | 19 | physical fault |
| 1549 | scan | 19 | normal |
| 36 | scan | 19 | normal |
| 1628 | normal | 18 | MITM |
| 5 | normal | 18 | MITM |
| 34 | normal | 18 | MITM |
| 35 | normal | 18 | MITM |
| 1104 | normal | 18 | MITM |
| 160 | normal | 17 | MITM |
| 99 | normal | 17 | MITM |
| 210 | normal | 17 | MITM |
| 598 | normal | 16 | scan |
| 864 | normal | 16 | scan |
| 466 | normal | 16 | DoS |
| 1203 | normal | 16 | scan |
| 432 | normal | 15 | MITM |
| 790 | normal | 15 | scan |

---

## 🔗 Model Error Correlation

Which models make similar mistakes? (Jaccard similarity of error sets)

**Correlation Matrix:**

| Model | mlp | transformer | knn | mlp | forest | transformer | xgboost | knn | knn | knn | knn | forest | xgboost | knn | forest | mlp | transformer | transformer | mlp |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | 1.00 | 0.45 | 0.18 | 0.68 | 0.20 | 0.33 | 0.30 | 0.19 | 0.50 | 0.60 | 0.19 | 0.21 | 0.07 | 0.19 | 0.21 | 0.70 | 0.22 | 0.54 | 0.66 |
| transformer | 0.45 | 1.00 | 0.11 | 0.43 | 0.32 | 0.20 | 0.33 | 0.12 | 0.32 | 0.43 | 0.12 | 0.14 | 0.04 | 0.12 | 0.14 | 0.44 | 0.13 | 0.46 | 0.37 |
| knn | 0.18 | 0.11 | 1.00 | 0.14 | 0.05 | 0.37 | 0.10 | 0.73 | 0.32 | 0.19 | 0.73 | 0.16 | 0.14 | 0.73 | 0.16 | 0.15 | 0.38 | 0.15 | 0.22 |
| mlp | 0.68 | 0.43 | 0.14 | 1.00 | 0.25 | 0.26 | 0.31 | 0.15 | 0.41 | 0.55 | 0.15 | 0.17 | 0.05 | 0.15 | 0.17 | 0.70 | 0.16 | 0.50 | 0.47 |
| forest | 0.20 | 0.32 | 0.05 | 0.25 | 1.00 | 0.07 | 0.33 | 0.05 | 0.13 | 0.21 | 0.05 | 0.08 | 0.02 | 0.05 | 0.08 | 0.25 | 0.05 | 0.25 | 0.13 |
| transformer | 0.33 | 0.20 | 0.37 | 0.26 | 0.07 | 1.00 | 0.16 | 0.43 | 0.37 | 0.26 | 0.43 | 0.36 | 0.14 | 0.43 | 0.36 | 0.24 | 0.52 | 0.25 | 0.47 |
| xgboost | 0.30 | 0.33 | 0.10 | 0.31 | 0.33 | 0.16 | 1.00 | 0.10 | 0.26 | 0.32 | 0.10 | 0.20 | 0.06 | 0.10 | 0.20 | 0.31 | 0.11 | 0.32 | 0.23 |
| knn | 0.19 | 0.12 | 0.73 | 0.15 | 0.05 | 0.43 | 0.10 | 1.00 | 0.33 | 0.18 | 1.00 | 0.20 | 0.15 | 1.00 | 0.20 | 0.15 | 0.49 | 0.16 | 0.27 |
| knn | 0.50 | 0.32 | 0.32 | 0.41 | 0.13 | 0.37 | 0.26 | 0.33 | 1.00 | 0.53 | 0.33 | 0.22 | 0.09 | 0.33 | 0.22 | 0.44 | 0.32 | 0.39 | 0.49 |
| knn | 0.60 | 0.43 | 0.19 | 0.55 | 0.21 | 0.26 | 0.32 | 0.18 | 0.53 | 1.00 | 0.18 | 0.17 | 0.05 | 0.18 | 0.17 | 0.61 | 0.19 | 0.48 | 0.48 |
| knn | 0.19 | 0.12 | 0.73 | 0.15 | 0.05 | 0.43 | 0.10 | 1.00 | 0.33 | 0.18 | 1.00 | 0.20 | 0.15 | 1.00 | 0.20 | 0.15 | 0.49 | 0.16 | 0.27 |
| forest | 0.21 | 0.14 | 0.16 | 0.17 | 0.08 | 0.36 | 0.20 | 0.20 | 0.22 | 0.17 | 0.20 | 1.00 | 0.20 | 0.20 | 1.00 | 0.16 | 0.28 | 0.17 | 0.23 |
| xgboost | 0.07 | 0.04 | 0.14 | 0.05 | 0.02 | 0.14 | 0.06 | 0.15 | 0.09 | 0.05 | 0.15 | 0.20 | 1.00 | 0.15 | 0.20 | 0.05 | 0.16 | 0.06 | 0.10 |
| knn | 0.19 | 0.12 | 0.73 | 0.15 | 0.05 | 0.43 | 0.10 | 1.00 | 0.33 | 0.18 | 1.00 | 0.20 | 0.15 | 1.00 | 0.20 | 0.15 | 0.49 | 0.16 | 0.27 |
| forest | 0.21 | 0.14 | 0.16 | 0.17 | 0.08 | 0.36 | 0.20 | 0.20 | 0.22 | 0.17 | 0.20 | 1.00 | 0.20 | 0.20 | 1.00 | 0.16 | 0.28 | 0.17 | 0.23 |
| mlp | 0.70 | 0.44 | 0.15 | 0.70 | 0.25 | 0.24 | 0.31 | 0.15 | 0.44 | 0.61 | 0.15 | 0.16 | 0.05 | 0.15 | 0.16 | 1.00 | 0.16 | 0.49 | 0.47 |
| transformer | 0.22 | 0.13 | 0.38 | 0.16 | 0.05 | 0.52 | 0.11 | 0.49 | 0.32 | 0.19 | 0.49 | 0.28 | 0.16 | 0.49 | 0.28 | 0.16 | 1.00 | 0.18 | 0.33 |
| transformer | 0.54 | 0.46 | 0.15 | 0.50 | 0.25 | 0.25 | 0.32 | 0.16 | 0.39 | 0.48 | 0.16 | 0.17 | 0.06 | 0.16 | 0.17 | 0.49 | 0.18 | 1.00 | 0.46 |
| mlp | 0.66 | 0.37 | 0.22 | 0.47 | 0.13 | 0.47 | 0.23 | 0.27 | 0.49 | 0.48 | 0.27 | 0.23 | 0.10 | 0.27 | 0.23 | 0.47 | 0.33 | 0.46 | 1.00 |

![Correlation Heatmap](correlation/correlation_heatmap.png)

---

## 📈 Training Curves

### 2026-01-14_21-23-07_physical_small_attention_mlp

![2026-01-14_21-23-07_physical_small_attention_mlp](training_curves/2026-01-14_21-23-07_physical_small_attention_mlp.png)

### 2026-01-14_22-00-45_physical_small_ft_transformer

![2026-01-14_22-00-45_physical_small_ft_transformer](training_curves/2026-01-14_22-00-45_physical_small_ft_transformer.png)

### 2026-01-14_23-01-46_physical_small_knn

![2026-01-14_23-01-46_physical_small_knn](training_curves/2026-01-14_23-01-46_physical_small_knn.png)

### 2026-01-14_23-14-32_physical_small_mlp

![2026-01-14_23-14-32_physical_small_mlp](training_curves/2026-01-14_23-14-32_physical_small_mlp.png)

### 2026-01-14_23-16-06_physical_small_random_forest

![2026-01-14_23-16-06_physical_small_random_forest](training_curves/2026-01-14_23-16-06_physical_small_random_forest.png)

### 2026-01-14_23-38-28_physical_small_tab_transformer

![2026-01-14_23-38-28_physical_small_tab_transformer](training_curves/2026-01-14_23-38-28_physical_small_tab_transformer.png)

### 2026-01-14_23-39-57_physical_small_xgboost

![2026-01-14_23-39-57_physical_small_xgboost](training_curves/2026-01-14_23-39-57_physical_small_xgboost.png)

### 2026-01-20_20-18-39_physical_small_k5_dist_knn

![2026-01-20_20-18-39_physical_small_k5_dist_knn](training_curves/2026-01-20_20-18-39_physical_small_k5_dist_knn.png)

### 2026-01-20_20-18-50_physical_small_k10_manhattan_knn

![2026-01-20_20-18-50_physical_small_k10_manhattan_knn](training_curves/2026-01-20_20-18-50_physical_small_k10_manhattan_knn.png)

### 2026-01-20_20-19-22_physical_small_k20_uniform_knn

![2026-01-20_20-19-22_physical_small_k20_uniform_knn](training_curves/2026-01-20_20-19-22_physical_small_k20_uniform_knn.png)

### 2026-01-20_20-26-50_physical_medium_knn

![2026-01-20_20-26-50_physical_medium_knn](training_curves/2026-01-20_20-26-50_physical_medium_knn.png)

### 2026-01-20_20-26-58_physical_medium_random_forest

![2026-01-20_20-26-58_physical_medium_random_forest](training_curves/2026-01-20_20-26-58_physical_medium_random_forest.png)

### 2026-01-20_20-35-18_physical_medium_xgboost

![2026-01-20_20-35-18_physical_medium_xgboost](training_curves/2026-01-20_20-35-18_physical_medium_xgboost.png)

### 2026-01-20_20-35-43_physical_medium_knn

![2026-01-20_20-35-43_physical_medium_knn](training_curves/2026-01-20_20-35-43_physical_medium_knn.png)

### 2026-01-20_20-35-51_physical_medium_random_forest

![2026-01-20_20-35-51_physical_medium_random_forest](training_curves/2026-01-20_20-35-51_physical_medium_random_forest.png)

### 2026-01-20_21-53-48_physical_medium_mlp

![2026-01-20_21-53-48_physical_medium_mlp](training_curves/2026-01-20_21-53-48_physical_medium_mlp.png)

### 2026-01-20_21-54-31_physical_medium_tab_transformer

![2026-01-20_21-54-31_physical_medium_tab_transformer](training_curves/2026-01-20_21-54-31_physical_medium_tab_transformer.png)

### 2026-01-20_21-58-46_physical_medium_ft_transformer

![2026-01-20_21-58-46_physical_medium_ft_transformer](training_curves/2026-01-20_21-58-46_physical_medium_ft_transformer.png)

### 2026-01-20_22-00-54_physical_medium_attention_mlp

![2026-01-20_22-00-54_physical_medium_attention_mlp](training_curves/2026-01-20_22-00-54_physical_medium_attention_mlp.png)

