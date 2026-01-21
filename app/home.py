import streamlit as st

st.title("Home")

st.header("Welcome to this ML model dashboard project.")


st.divider()

# Introduction
st.header("About This Project")
st.markdown("""
This dashboard provides comprehensive insights into machine learning models trained for 
**Network Intrusion Detection**. The project analyzes both network traffic data and physical 
sensor data to identify potential security threats and anomalies.

The models have been trained to classify different types of network attacks and normal traffic 
patterns, helping to understand model performance across various evaluation metrics.
""")

st.divider()

st.header("Structure")


st.subheader("Data Overview")
st.markdown("""
- **Network Data**: Traffic patterns, protocols, IP addresses
- **Physical Data**: Sensor readings and anomaly detection
- **Label Distribution**: Attack types / normal traffic
- **Missing Values**: Data quality analysis
""")

st.subheader("Model Performance")
st.markdown("""
- **Training History**: Loss and accuracy progression
- **Performance Metrics**: Accuracy, F1-score, MCC, precision, recall
- **Per-Class Analysis**: How well each attack type is detected
- **Confusion Matrices**: Detailed classification results
- **Cross-validation**: Train, validation, and test set performance
""")

st.subheader("Training Insights")
st.markdown("""
- **Epoch Progression**: Training and validation curves
- **Convergence Patterns**: How quickly models learn
""")