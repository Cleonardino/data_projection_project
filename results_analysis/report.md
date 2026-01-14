# ML Experiment Results Analysis

Generated: 2026-01-14 19:55:31

---

## 📊 Model Leaderboard

Ranked by F1 (macro) score on test set.

| Rank | Model | Dataset | Config | Accuracy | F1 (macro) | Balanced Acc | MCC | Time (s) |
|------|---|---|---|---|---|---|---|---|
| 1 | xgboost | physical | physical_small | 0.9100 | 0.7723 | 0.7286 | 0.7588 | 0.3 |

---

## 🎯 Hard Classes Analysis

Classes with highest error rates across all models.

| Class | Total | Errors | Error Rate |
|-------|-------|--------|------------|
| nomal | 6 | 4 | 66.67% |
| MITM | 31 | 12 | 38.71% |
| DoS | 11 | 2 | 18.18% |
| physical fault | 22 | 2 | 9.09% |
| normal | 230 | 7 | 3.04% |

---

## 🔍 Hardest Samples

Samples misclassified by most models (top 20).

| Sample | True Label | Times Wrong | Common Prediction |
|--------|------------|-------------|-------------------|
| 12 | DoS | 1 | normal |
| 13 | physical fault | 1 | normal |
| 35 | MITM | 1 | normal |
| 51 | normal | 1 | MITM |
| 64 | MITM | 1 | normal |
| 68 | normal | 1 | DoS |
| 89 | MITM | 1 | normal |
| 91 | MITM | 1 | normal |
| 102 | MITM | 1 | normal |
| 106 | MITM | 1 | normal |
| 119 | MITM | 1 | normal |
| 121 | normal | 1 | nomal |
| 145 | MITM | 1 | normal |
| 153 | nomal | 1 | normal |
| 166 | DoS | 1 | normal |
| 173 | normal | 1 | MITM |
| 177 | nomal | 1 | normal |
| 181 | MITM | 1 | normal |
| 184 | physical fault | 1 | normal |
| 208 | normal | 1 | DoS |

---

## 🔗 Model Error Correlation

Which models make similar mistakes? (Jaccard similarity of error sets)

*Need at least 2 experiments for correlation analysis.*

---

## 📈 Training Curves

*No training curves available.*
