# Analysis Report: Network Dataset

Generated: 2026-01-15 09:51:48

---

## 📊 Model Leaderboard

Ranked by F1 (macro) score on test set.

| Rank | Experiment Name | Accuracy | F1 (macro) | Balanced Acc | MCC | Time (s) |
|------|---|---|---|---|---|---|
| 1 | `2026-01-14_23-14-58_network_small_random_forest` | 0.8760 | 0.8688 | 0.9592 | 0.5627 | 43.0 |
| 2 | `2026-01-15_00-55-56_network_small_knn` | 0.8990 | 0.8368 | 0.8442 | 0.3495 | 5.3 |
| 3 | `2026-01-15_00-11-13_network_small_ft_transformer` | 0.8552 | 0.7743 | 0.9598 | 0.5401 | 2673.9 |
| 4 | `2026-01-14_23-01-53_network_small_mlp` | 0.8510 | 0.7718 | 0.9585 | 0.5333 | 745.1 |
| 5 | `2026-01-15_01-05-22_network_small_tab_transformer` | 0.4656 | 0.7042 | 0.8548 | 0.2300 | 2786.7 |
| 6 | `2026-01-15_09-41-15_network_medium_xgboost` | 0.7113 | 0.6961 | 0.9279 | 0.4693 | 122.6 |
| 7 | `2026-01-15_09-37-40_network_medium_random_forest` | 0.7058 | 0.6843 | 0.8724 | 0.4665 | 165.7 |
| 8 | `2026-01-14_23-39-12_network_small_xgboost` | 0.8663 | 0.6134 | 0.7095 | 0.5515 | 26.5 |
| 9 | `2026-01-14_23-49-15_network_small_attention_mlp` | 0.6476 | 0.5953 | 0.8121 | 0.1380 | 1308.3 |
| 10 | `2026-01-15_09-26-36_network_medium_knn` | 0.8589 | 0.5909 | 0.5706 | 0.2198 | 9.9 |

---

## 🎯 Hard Classes Analysis

Classes with highest error rates across all models.

| Class | Total | Errors | Error Rate |
|-------|-------|--------|------------|
| MITM | 80409 | 19950 | 24.81% |
| normal | 2442511 | 562822 | 23.04% |
| physical fault | 174090 | 35001 | 20.11% |
| anomaly | 112 | 14 | 12.50% |
| scan | 16 | 2 | 12.50% |
| DoS | 2862 | 1 | 0.03% |

---

## 🔍 Hardest Samples

Samples misclassified by most models (top 20).

| Sample | True Label | Times Wrong | Common Prediction |
|--------|------------|-------------|-------------------|
| 224542 | normal | 10 | physical fault |
| 123286 | normal | 10 | physical fault |
| 28766 | normal | 10 | physical fault |
| 93924 | normal | 10 | physical fault |
| 28704 | normal | 10 | physical fault |
| 82777 | normal | 10 | physical fault |
| 22874 | normal | 10 | physical fault |
| 70104 | normal | 10 | physical fault |
| 150072 | normal | 10 | physical fault |
| 122844 | normal | 10 | physical fault |
| 122707 | normal | 10 | physical fault |
| 197763 | normal | 10 | physical fault |
| 133546 | normal | 10 | physical fault |
| 109816 | normal | 10 | physical fault |
| 189150 | normal | 10 | physical fault |
| 104560 | normal | 10 | physical fault |
| 99694 | physical fault | 10 | physical fault |
| 104099 | physical fault | 10 | physical fault |
| 150768 | normal | 10 | physical fault |
| 140389 | normal | 10 | physical fault |

---

## 🔗 Model Error Correlation

Which models make similar mistakes? (Jaccard similarity of error sets)

**Correlation Matrix:**

| Model | mlp | forest | xgboost | mlp | transformer | knn | transformer | knn | forest | xgboost |
|-------|---|---|---|---|---|---|---|---|---|---|
| mlp | 1.00 | 0.79 | 0.85 | 0.20 | 0.92 | 0.28 | 0.27 | 0.06 | 0.07 | 0.07 |
| forest | 0.79 | 1.00 | 0.88 | 0.18 | 0.81 | 0.31 | 0.23 | 0.05 | 0.06 | 0.06 |
| xgboost | 0.85 | 0.88 | 1.00 | 0.19 | 0.88 | 0.30 | 0.25 | 0.06 | 0.07 | 0.07 |
| mlp | 0.20 | 0.18 | 0.19 | 1.00 | 0.20 | 0.11 | 0.39 | 0.09 | 0.14 | 0.14 |
| transformer | 0.92 | 0.81 | 0.88 | 0.20 | 1.00 | 0.29 | 0.27 | 0.06 | 0.07 | 0.07 |
| knn | 0.28 | 0.31 | 0.30 | 0.11 | 0.29 | 1.00 | 0.10 | 0.05 | 0.05 | 0.05 |
| transformer | 0.27 | 0.23 | 0.25 | 0.39 | 0.27 | 0.10 | 1.00 | 0.11 | 0.19 | 0.18 |
| knn | 0.06 | 0.05 | 0.06 | 0.09 | 0.06 | 0.05 | 0.11 | 1.00 | 0.12 | 0.12 |
| forest | 0.07 | 0.06 | 0.07 | 0.14 | 0.07 | 0.05 | 0.19 | 0.12 | 1.00 | 0.97 |
| xgboost | 0.07 | 0.06 | 0.07 | 0.14 | 0.07 | 0.05 | 0.18 | 0.12 | 0.97 | 1.00 |

![Correlation Heatmap](correlation/correlation_heatmap.png)

---

## 📈 Training Curves

### 2026-01-14_23-01-53_network_small_mlp

![2026-01-14_23-01-53_network_small_mlp](training_curves/2026-01-14_23-01-53_network_small_mlp.png)

### 2026-01-14_23-14-58_network_small_random_forest

![2026-01-14_23-14-58_network_small_random_forest](training_curves/2026-01-14_23-14-58_network_small_random_forest.png)

### 2026-01-14_23-39-12_network_small_xgboost

![2026-01-14_23-39-12_network_small_xgboost](training_curves/2026-01-14_23-39-12_network_small_xgboost.png)

### 2026-01-14_23-49-15_network_small_attention_mlp

![2026-01-14_23-49-15_network_small_attention_mlp](training_curves/2026-01-14_23-49-15_network_small_attention_mlp.png)

### 2026-01-15_00-11-13_network_small_ft_transformer

![2026-01-15_00-11-13_network_small_ft_transformer](training_curves/2026-01-15_00-11-13_network_small_ft_transformer.png)

### 2026-01-15_00-55-56_network_small_knn

![2026-01-15_00-55-56_network_small_knn](training_curves/2026-01-15_00-55-56_network_small_knn.png)

### 2026-01-15_01-05-22_network_small_tab_transformer

![2026-01-15_01-05-22_network_small_tab_transformer](training_curves/2026-01-15_01-05-22_network_small_tab_transformer.png)

### 2026-01-15_09-26-36_network_medium_knn

![2026-01-15_09-26-36_network_medium_knn](training_curves/2026-01-15_09-26-36_network_medium_knn.png)

### 2026-01-15_09-37-40_network_medium_random_forest

![2026-01-15_09-37-40_network_medium_random_forest](training_curves/2026-01-15_09-37-40_network_medium_random_forest.png)

### 2026-01-15_09-41-15_network_medium_xgboost

![2026-01-15_09-41-15_network_medium_xgboost](training_curves/2026-01-15_09-41-15_network_medium_xgboost.png)

