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
Explore the datasets used for training and testing:
- **Network Data**: Traffic patterns, protocols, IP addresses
- **Physical Data**: Sensor readings and anomaly detection
- **Label Distribution**: Attack types / normal traffic
- **Temporal Patterns**: How events are distributed over time
- **Missing Values**: Data quality analysis
""")

st.subheader("Model Performance")
st.markdown("""
Analyze trained models with detailed metrics:
- **Training History**: Loss and accuracy progression
- **Performance Metrics**: Accuracy, F1-score, MCC, precision, recall
- **Per-Class Analysis**: How well each attack type is detected
- **Confusion Matrices**: Detailed classification results
- **Cross-validation**: Train, validation, and test set performance
""")

st.subheader("Training Insights")
st.markdown("""
Understand how models were trained:
- **Epoch Progression**: Training and validation curves
- **Best Checkpoints**: Optimal stopping points
- **Overfitting Detection**: Training vs. validation gaps
- **Convergence Patterns**: How quickly models learn
""")

st.subheader("Model Comparison")
st.markdown("""
Compare different models side-by-side:
- **Architecture Variations**: Small, medium, large networks
- **Algorithm Differences**: Neural networks vs. XGBoost
- **Dataset Impact**: Network-only vs. combined datasets
- **Time Analysis**: Training efficiency and speed
""")

st.divider()

# How to use
st.header("🚀 How to Use This Dashboard")

st.markdown("""
1. **Navigation**: Use the sidebar to switch between different sections
   - **General**: Home, Data Overview, and Contact pages
   - **Models**: Individual model performance pages (organized by date and name)

2. **Data Overview**: Start here to understand the datasets
   - Review global statistics and data distribution
   - Analyze attack patterns and protocol usage
   - Examine temporal trends and missing values

3. **Model Pages**: Dive into specific model results
   - Each page shows training history, metrics, and detailed performance
   - Use tabs to switch between train, validation, and test results
   - Expand tables to see full training histories

4. **Comparison**: Navigate between different models to compare
   - Note the date in the subtitle to track experiments chronologically
   - Compare metrics across different architectures and configurations
""")

st.divider()

# Key metrics explanation
st.header("📚 Understanding the Metrics")

with st.expander("🎯 Classification Metrics Explained"):
    st.markdown("""
    **Accuracy**: Overall percentage of correct predictions
    - Good for balanced datasets
    - Can be misleading with imbalanced classes
    
    **Balanced Accuracy**: Average of recall obtained on each class
    - Better for imbalanced datasets
    - Treats all classes equally
    
    **F1-Score**: Harmonic mean of precision and recall
    - Balances false positives and false negatives
    - Macro: Average across all classes
    - Weighted: Accounts for class imbalance
    
    **MCC (Matthews Correlation Coefficient)**: Correlation between predictions and truth
    - Ranges from -1 to +1
    - Considers true/false positives and negatives
    - Robust metric for imbalanced datasets
    
    **Precision**: Of all positive predictions, how many were correct?
    - Important when false positives are costly
    
    **Recall**: Of all actual positives, how many were detected?
    - Important when false negatives are costly
    """)

with st.expander("📊 Training Metrics Explained"):
    st.markdown("""
    **Loss**: Measure of how wrong the model's predictions are
    - Lower is better
    - Should decrease over epochs
    
    **Training Loss vs. Validation Loss**:
    - Similar values: Good generalization
    - Val loss > Train loss: Potential overfitting
    - Val loss increasing: Definite overfitting
    
    **Accuracy**: Percentage of correct predictions
    - Should increase over epochs
    - Monitor gap between train and validation
    
    **Best Epoch**: When validation performance was optimal
    - Models are saved at this checkpoint
    - Prevents overfitting to training data
    """)

st.divider()

# Dataset information
st.header("📦 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Network Dataset")
    st.markdown("""
    Contains network traffic logs with features such as:
    - Source and destination IP addresses
    - Protocol types (TCP, UDP, ICMP, etc.)
    - Port numbers and packet information
    - Traffic flow statistics
    - Attack labels (DDoS, PortScan, Infiltration, etc.)
    """)

with col2:
    st.subheader("Physical Dataset")
    st.markdown("""
    Contains sensor readings from IoT devices:
    - Temperature, humidity, motion sensors
    - Timestamp information
    - Sensor activation patterns
    - Anomaly labels
    - Multi-sensor correlation data
    """)

st.divider()

# Footer
st.header("📞 Need Help?")
st.markdown("""
For questions, issues, or suggestions:
- Visit the **Contact** page in the sidebar
- Review model-specific documentation on each model page
- Check the **Data Overview** for dataset-level insights

Happy exploring! 🎉
""")

st.divider()

# Quick stats teaser
st.info("💡 **Quick Tip**: Start with the Data Overview page to get familiar with the datasets before diving into model results!")