# Analysis Report: Network Dataset

Generated: 2026-01-14 23:56:34

---

## 📊 Model Leaderboard

Ranked by F1 (macro) score on test set.

| Rank | Model | Config | Accuracy | F1 (macro) | Balanced Acc | MCC | Time (s) |
|------|---|---|---|---|---|---|---|
| 1 | forest | network_small | 0.8760 | 0.8688 | 0.9592 | 0.5627 | 43.0 |
| 2 | mlp | network_small | 0.8510 | 0.7718 | 0.9585 | 0.5333 | 745.1 |
| 3 | xgboost | network_small | 0.8663 | 0.6134 | 0.7095 | 0.5515 | 26.5 |

---

## 🎯 Hard Classes Analysis

Classes with highest error rates across all models.

| Class | Total | Errors | Error Rate |
|-------|-------|--------|------------|
| scan | 3 | 1 | 33.33% |
| normal | 622740 | 90545 | 14.54% |
| physical fault | 52227 | 964 | 1.85% |
| anomaly | 30 | 0 | 0.00% |

---

## 🔍 Hardest Samples

Samples misclassified by most models (top 20).

| Sample | True Label | Times Wrong | Common Prediction |
|--------|------------|-------------|-------------------|
| 224998 | normal | 3 | physical fault |
| 224989 | normal | 3 | physical fault |
| 224985 | normal | 3 | physical fault |
| 224983 | normal | 3 | physical fault |
| 224979 | normal | 3 | physical fault |
| 224959 | normal | 3 | physical fault |
| 224954 | normal | 3 | physical fault |
| 224948 | normal | 3 | physical fault |
| 224945 | normal | 3 | physical fault |
| 224941 | normal | 3 | physical fault |
| 224936 | normal | 3 | physical fault |
| 224921 | normal | 3 | physical fault |
| 261 | normal | 3 | physical fault |
| 259 | normal | 3 | physical fault |
| 241 | normal | 3 | physical fault |
| 239 | normal | 3 | physical fault |
| 224 | normal | 3 | physical fault |
| 203 | normal | 3 | physical fault |
| 200 | normal | 3 | physical fault |
| 199 | normal | 3 | physical fault |

---

## 🔗 Model Error Correlation

Which models make similar mistakes? (Jaccard similarity of error sets)

**Correlation Matrix:**

| Model | mlp | forest | xgboost |
|-------|---|---|---|
| mlp | 1.00 | 0.81 | 0.86 |
| forest | 0.81 | 1.00 | 0.89 |
| xgboost | 0.86 | 0.89 | 1.00 |


---

## 📈 Training Curves

*No training curves available.*
