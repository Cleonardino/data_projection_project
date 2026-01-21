import streamlit as st
import numpy as np
import pandas as pd
from src.summaries import load_experiments

class ModelData:
    def __init__(self, title : str, icon : str, metrics, train_history):
        self.title = title
        self.icon = icon
        self.metrics = metrics
        self.train_history = train_history
    
    def page_function(self):
        st.title(self.title)
        
        # Training Overview
        st.header("📊 Training Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Epochs", self.metrics["training"]["total_epochs"])
        with col2:
            st.metric("Best Epoch", self.metrics["training"]["best_epoch"])
        with col3:
            st.metric("Training Time", f"{self.metrics['training']['training_time_seconds']:.2f}s")
        
        # Training History
        st.header("📈 Training History")
        
        # Create DataFrame from training history
        history_df = pd.DataFrame({
            'Epoch': self.train_history['epochs'],
            'Train Loss': self.train_history['train_loss'],
            'Val Loss': self.train_history['val_loss'],
            'Train Accuracy': self.train_history['train_accuracy'],
            'Val Accuracy': self.train_history['val_accuracy']
        })
        
        # Plot Loss
        st.subheader("Loss")
        loss_df = history_df[['Epoch', 'Train Loss', 'Val Loss']].set_index('Epoch')
        st.line_chart(loss_df)
        
        # Plot Accuracy
        st.subheader("Accuracy")
        acc_df = history_df[['Epoch', 'Train Accuracy', 'Val Accuracy']].set_index('Epoch')
        st.line_chart(acc_df)
        
        # Show table with option to expand
        with st.expander("View Training History Table"):
            st.dataframe(history_df, use_container_width=True)
        
        # Metrics Section
        st.header("🎯 Model Metrics")
        
        # Create tabs for train/val/test
        tab1, tab2, tab3 = st.tabs(["Train", "Validation", "Test"])
        
        for tab, dataset in zip([tab1, tab2, tab3], ['train', 'val', 'test']):
            with tab:
                metrics_data = self.metrics['metrics'][dataset]
                
                # Overall metrics
                st.subheader("Overall Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics_data['accuracy']:.4f}")
                with col2:
                    st.metric("Balanced Accuracy", f"{metrics_data['balanced_accuracy']:.4f}")
                with col3:
                    st.metric("F1 (Macro)", f"{metrics_data['f1_macro']:.4f}")
                with col4:
                    st.metric("MCC", f"{metrics_data['mcc']:.4f}")
                
                # Detailed metrics table
                st.subheader("Detailed Metrics")
                detailed_metrics_df = pd.DataFrame({
                    'Precision (Macro)': [metrics_data['precision_macro']],
                    'Recall (Macro)': [metrics_data['recall_macro']],
                    'Precision (Weighted)': [metrics_data['precision_weighted']],
                    'Recall (Weighted)': [metrics_data['recall_weighted']],
                    'F1 (Weighted)': [metrics_data['f1_weighted']]
                })
                st.dataframe(detailed_metrics_df, use_container_width=True)
                
                # Per-class metrics
                st.subheader("Per-Class Metrics")
                num_classes = len(metrics_data['per_class_precision'])
                per_class_df = pd.DataFrame({
                    'Class': [f'Class {i}' for i in range(num_classes)],
                    'Precision': metrics_data['per_class_precision'],
                    'Recall': metrics_data['per_class_recall'],
                    'F1-Score': metrics_data['per_class_f1']
                })
                st.dataframe(per_class_df, use_container_width=True)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm_df = pd.DataFrame(
                    metrics_data['confusion_matrix'],
                    columns=[f'Pred {i}' for i in range(num_classes)],
                    index=[f'True {i}' for i in range(num_classes)]
                )
                st.dataframe(cm_df, use_container_width=True)

model_data : list[ModelData] = []

experiments = load_experiments()
print(len(experiments))

for key in experiments:
    # Create page for the experiment
    model_data.append(ModelData(
        key,
        ":material/home:",
        experiments[key]["metrics"],
        experiments[key]["history"]
        ))

model_pages : list = []

for model in model_data:
    model_pages.append(st.Page(
        model.page_function,
        title=model.title,
        icon=":material/keyboard_double_arrow_right:",
        url_path=model.title
        ))

navigation = st.navigation(
	{
		"General": [
			st.Page("app/home.py", title="Home", icon=":material/home:"),
   			st.Page("app/data_overview.py", title="Data Overview", icon=":material/dataset:"),
            st.Page("app/contact.py", title="Contact", icon=":material/contact_page:")
		],
		"Models": model_pages
	}
)

navigation.run()

# https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded
# https://docs.streamlit.io/develop/tutorials/multipage/dynamic-navigation

# dataframe = np.random.randn(10, 20)
# st.dataframe(dataframe)

# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))

# st.dataframe(dataframe.style.highlight_max(axis=0))

# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)

# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.map(map_data)

# x = st.slider('x')  # 👈 this is a widget
# st.write(x, 'squared is', x * x)

# st.text_input("Your name", key="name")

# # You can access the value at any point with:
# st.session_state.name

# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])

#     chart_data

# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
#     })

# option = st.selectbox(
#     'Which number do you like best?',
#      df['first column'])

# 'You selected: ', option

# # Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

# # Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )

# left_column, right_column = st.columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')

# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")

# import time

# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'