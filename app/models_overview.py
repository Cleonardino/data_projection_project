import streamlit as st

st.title("Models Overview")

st.write("To look at the precise data of each model, you can select it. Here is presented the correlation of each model type, even ones which weren't kept")
st.write("The models presented are the ones which were kept due to sufficient metrics and interesting training data.")

st.header("Physical Model Correlation")
st.image("results_analysis/physical/correlation/correlation_heatmap.png")

st.header("Network Model Correlation")
st.image("results_analysis/network/correlation/correlation_heatmap.png")