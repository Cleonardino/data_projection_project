# Analysis Report: Network Dataset

Generated: 2026-01-15 07:25:35

---

## 📊 Model Leaderboard

Ranked by F1 (macro) score on test set.

| Rank | Experiment Name                                     | Accuracy | F1 (macro) | Balanced Acc | MCC    | Time (s) |
| ---- | --------------------------------------------------- | -------- | ---------- | ------------ | ------ | -------- |
| 1    | `2026-01-14_23-14-58_network_small_random_forest`   | 0.8760   | 0.8688     | 0.9592       | 0.5627 | 43.0     |
| 2    | `2026-01-15_00-55-56_network_small_knn`             | 0.8990   | 0.8368     | 0.8442       | 0.3495 | 5.3      |
| 3    | `2026-01-15_00-11-13_network_small_ft_transformer`  | 0.8552   | 0.7743     | 0.9598       | 0.5401 | 2673.9   |
| 4    | `2026-01-14_23-01-53_network_small_mlp`             | 0.8510   | 0.7718     | 0.9585       | 0.5333 | 745.1    |
| 5    | `2026-01-15_01-05-22_network_small_tab_transformer` | 0.4656   | 0.7042     | 0.8548       | 0.2300 | 2786.7   |
| 6    | `2026-01-14_23-39-12_network_small_xgboost`         | 0.8663   | 0.6134     | 0.7095       | 0.5515 | 26.5     |
| 7    | `2026-01-14_23-49-15_network_small_attention_mlp`   | 0.6476   | 0.5953     | 0.8121       | 0.1380 | 1308.3   |

---

## 🎯 Hard Classes Analysis

Classes with highest error rates across all models.

| Class          | Total   | Errors | Error Rate |
| -------------- | ------- | ------ | ---------- |
| normal         | 1453060 | 328475 | 22.61%     |
| physical fault | 121863  | 17847  | 14.65%     |
| scan           | 7       | 1      | 14.29%     |
| anomaly        | 70      | 0      | 0.00%      |

---

## 🔍 Hardest Samples

Samples misclassified by most models (top 20).

| Sample | True Label | Times Wrong | Common Prediction |
| ------ | ---------- | ----------- | ----------------- |
| 192    | normal     | 7           | physical fault    |
| 306    | normal     | 7           | physical fault    |
| 191992 | normal     | 7           | physical fault    |
| 191640 | normal     | 7           | physical fault    |
| 191641 | normal     | 7           | physical fault    |
| 191893 | normal     | 7           | physical fault    |
| 451    | normal     | 7           | physical fault    |
| 276    | normal     | 7           | physical fault    |
| 15     | normal     | 7           | physical fault    |
| 18     | normal     | 7           | physical fault    |
| 27     | normal     | 7           | physical fault    |
| 191760 | normal     | 7           | physical fault    |
| 191794 | normal     | 7           | physical fault    |
| 191825 | normal     | 7           | physical fault    |
| 191663 | normal     | 7           | physical fault    |
| 191675 | normal     | 7           | physical fault    |
| 191455 | normal     | 7           | physical fault    |
| 191519 | normal     | 7           | physical fault    |
| 191267 | normal     | 7           | physical fault    |
| 426    | normal     | 7           | physical fault    |

---

## 🔗 Model Error Correlation

Which models make similar mistakes? (Jaccard similarity of error sets)

**Correlation Matrix:**

| Model       | mlp  | forest | xgboost | mlp  | transformer | knn  | transformer |
| ----------- | ---- | ------ | ------- | ---- | ----------- | ---- | ----------- |
| mlp         | 1.00 | 0.81   | 0.86    | 0.21 | 0.92        | 0.30 | 0.27        |
| forest      | 0.81 | 1.00   | 0.89    | 0.19 | 0.82        | 0.33 | 0.23        |
| xgboost     | 0.86 | 0.89   | 1.00    | 0.20 | 0.89        | 0.32 | 0.25        |
| mlp         | 0.21 | 0.19   | 0.20    | 1.00 | 0.21        | 0.11 | 0.39        |
| transformer | 0.92 | 0.82   | 0.89    | 0.21 | 1.00        | 0.31 | 0.27        |
| knn         | 0.30 | 0.33   | 0.32    | 0.11 | 0.31        | 1.00 | 0.10        |
| transformer | 0.27 | 0.23   | 0.25    | 0.39 | 0.27        | 0.10 | 1.00        |

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
