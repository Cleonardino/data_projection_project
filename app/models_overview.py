import streamlit as st
import pandas as pd
import numpy as np
from src.summaries import load_experiments

st.title("Models Overview")

st.write("To look at the precise data of each model, you can select it. Here is presented the correlation of each model type, even ones which weren't kept")
st.write("The models presented are the ones which were kept due to sufficient metrics and interesting training data.")

st.divider()

# ============================================================================
# LEADERBOARD SECTION
# ============================================================================
st.header("🏆 Models Leaderboard")

# Load experiments
experiments = load_experiments()

# Extract model data for leaderboard
leaderboard_data = []

for model_name, model_info in experiments.items():
    # Parse model name to extract readable parts
    parts = model_name.split('_')
    if len(parts) >= 3:
        date = parts[0]  # 2026-01-14
        model_type = '_'.join(parts[3:])  # network_small_xgboost
    else:
        date = "Unknown"
        model_type = model_name
    
    # Extract test metrics (most important for comparison)
    test_metrics = model_info["metrics"]["metrics"]["test"]
    training_info = model_info["metrics"]["training"]
    
    leaderboard_data.append({
        'Model Name': model_type,
        'Date': date,
        'Test Accuracy': test_metrics['accuracy'],
        'Test Balanced Accuracy': test_metrics['balanced_accuracy'],
        'Test F1 (Macro)': test_metrics['f1_macro'],
        'Test F1 (Weighted)': test_metrics['f1_weighted'],
        'Test Precision (Macro)': test_metrics['precision_macro'],
        'Test Recall (Macro)': test_metrics['recall_macro'],
        'Test MCC': test_metrics['mcc'],
        'Val Accuracy': model_info["metrics"]["metrics"]["val"]['accuracy'],
        'Val F1 (Macro)': model_info["metrics"]["metrics"]["val"]['f1_macro'],
        'Train Accuracy': model_info["metrics"]["metrics"]["train"]['accuracy'],
        'Total Epochs': training_info['total_epochs'],
        'Best Epoch': training_info['best_epoch'],
        'Training Time (s)': training_info['training_time_seconds']
    })

# Create DataFrame
df_leaderboard = pd.DataFrame(leaderboard_data)

# Metric selection
col1, col2 = st.columns([2, 1])

with col1:
    sort_metric = st.selectbox(
        "Select metric to sort by",
        options=[
            'Test Accuracy',
            'Test Balanced Accuracy',
            'Test F1 (Macro)',
            'Test F1 (Weighted)',
            'Test Precision (Macro)',
            'Test Recall (Macro)',
            'Test MCC',
            'Val Accuracy',
            'Val F1 (Macro)',
            'Train Accuracy',
            'Training Time (s)'
        ],
        index=0
    )

with col2:
    sort_order = st.radio(
        "Sort order",
        options=['Descending', 'Ascending'],
        index=0,
        horizontal=True
    )

# Sort the dataframe
ascending = sort_order == 'Ascending'
# For training time, lower is better, so invert the logic
if sort_metric == 'Training Time (s)':
    ascending = not ascending

df_sorted = df_leaderboard.sort_values(by=sort_metric, ascending=ascending)

# Display the dataframe
st.dataframe(df_sorted, use_container_width=True, height=500)

# Download button
csv = df_sorted.to_csv(index=False)
st.download_button(
    label="📥 Download as CSV",
    data=csv,
    file_name=f"models_leaderboard_{sort_metric.replace(' ', '_')}.csv",
    mime="text/csv"
)

st.divider()

# ============================================================================
# CORRELATION HEATMAPS
# ============================================================================
st.header("📈 Feature Correlation Analysis")

st.header("Physical Model Correlation")
st.image("results_analysis/physical/correlation/correlation_heatmap.png")

st.header("Network Model Correlation")
st.image("results_analysis/network/correlation/correlation_heatmap.png")