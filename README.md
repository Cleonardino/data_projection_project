# Cyber-Physical Systems Benchmarking

A modular benchmarking framework for analyzing machine learning models on Cyber-Physical System (CPS) datasets. Supports both physical sensor data and network traffic data.

## Project Structure

```
├── src/                    # Source code
│   ├── models/             # Neural network and ML model implementations
│   ├── lib_data.py         # Data loading, preprocessing, and caching
│   ├── train.py            # Single model training script
│   ├── train_all.py        # Batch training script
│   ├── generate_configs.py # Configuration generator
│   └── analyze_results.py  # automated analysis and reporting
├── configs/                # Generated YAML configuration files
├── dataset/                # Raw data directory
├── experiments/            # Training outputs (logs, models, metrics)
└── cached_datasets/        # Cached preprocessed data (pickle files)
```

## Setup

1. **Dependencies**: Ensure the following Python packages are installed:
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `torch`
   - `tqdm`
   - `pyyaml`
   - `xgboost` (optional, for XGBoost model)
   - `matplotlib`, `seaborn` (optional, for analysis plots)

2. **Data**: Place your datasets in the `dataset/` directory:
   - Physical data: `dataset/Physical dataset/*.csv`
   - Network data: `dataset/Network datatset/csv/*.csv`

## Workflow

### 1. Configuration Generation
Generate YAML config files for all models, datasets, and sizes (small/medium/large).
```bash
python src/generate_configs.py
```

### 2. Training
**Train a single model:**
```bash
python src/train.py --config configs/mlp/physical_small.yaml --model mlp
```

**Train batch of models:**
Run all experiments, or filter by model/dataset/size.
```bash
# Train everything
python src/train_all.py

# Filter specific subsets
python src/train_all.py --dataset physical --size small
python src/train_all.py --model xgboost mlp
```

### 3. Evaluation & Analysis
**Test a specific experiment:**
```bash
python src/test.py --experiment experiments/<experiment_timestamp_folder>
```

**Generate Analysis Report:**
Aggregates results, builds a leaderboard, and plots training curves.
```bash
python src/analyze_results.py
```
Outputs are saved to `results_analysis/` (report.md, leaderboard.csv, plots).

## Features

- **Data Caching**: Processed datasets are cached to `cached_datasets/` with a hash of their configuration. Subsequent runs load instantly.
- **Stratified Sampling**: Network datasets (large) are sampled using a memory-efficient stratified approach to ensure class balance (especially for rare 'anomaly' classes).
- **Progress Tracking**: 
  - Data loading shows detailed steps (loading, sampling, encoding).
  - Training loops show batch-level `tqdm` progress bars with live metrics.
- **Models Implemented**:
  - **Deep Learning**: `MLP`, `AttentionMLP`, `FT-Transformer`, `TabTransformer`
  - **Machine Learning**: `XGBoost`, `RandomForest`, `KNN`
- **Output Statistics**: `load_ml_ready_data` automatically prints class distribution statistics for Train, Validation, and Test splits.
