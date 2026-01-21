import streamlit as st
import pandas as pd
import numpy as np
from src.summaries import load_experiments

st.title("Models Overview")

st.write("To look at the precise data of each model, you can select it. Here is presented the correlation of each model type, even ones which weren't kept")
st.write("The models presented are the ones which were kept due to sufficient metrics and interesting training data.")

st.divider()

st.header("Models Leaderboard")
st.write("To sort by a particular metric, click on the corresponding column's name")

experiments = load_experiments()

network_data = []
physical_data = []

for model_name, model_info in experiments.items():
    parts = model_name.split('_')
    if len(parts) >= 3:
        date = parts[0]
        model_real_name = '_'.join(parts[2:])
    else:
        date = "Unknown"
        model_real_name = model_name
    
    # Get metrics
    test_metrics = model_info["metrics"]["metrics"]["test"]
    training_info = model_info["metrics"]["training"]
    
    model_data = {
        'Model Name': model_real_name,
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
    }
    
    # Split on type
    if model_real_name.startswith('network'):
        network_data.append(model_data)
    elif model_real_name.startswith('physical'):
        physical_data.append(model_data)

df_network = pd.DataFrame(network_data)
df_physical = pd.DataFrame(physical_data)

tab1, tab2 = st.tabs(["Network Models", "Physical Models"])

with tab1:
    if len(df_network) > 0:
        st.dataframe(df_network, use_container_width=True)
    else:
        st.info("No network models found")

with tab2:
    if len(df_physical) > 0:
        st.dataframe(df_physical, use_container_width=True)
    else:
        st.info("No physical models found")

st.divider()

st.header("Feature Correlation Analysis")

st.header("Physical Model Correlation")
st.image("results_analysis/physical/correlation/correlation_heatmap.png")

st.header("Network Model Correlation")
st.image("results_analysis/network/correlation/correlation_heatmap.png")