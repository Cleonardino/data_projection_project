import streamlit as st
import pandas as pd

class ModelData:
    def __init__(self, title: str, icon: str, metrics, train_history):
        self.title = title
        self.icon = icon
        self.metrics = metrics
        self.train_history = train_history
    
    def page_function(self):
        parts = self.title.split('_')
        # Parsing the title
        if len(parts) >= 3:
            date = parts[0]
            model_name = '_'.join(parts[3:])
            st.title(model_name)
            st.caption(date)
        else:
            st.title(self.title)
        
        # Training Overview
        st.header("Training")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Epochs Count", self.metrics["training"]["total_epochs"])
        with col2:
            st.metric("Best Epoch", self.metrics["training"]["best_epoch"])
        with col3:
            st.metric("Training Time", f"{self.metrics['training']['training_time_seconds']:.2f}s")
        
        # Training History
        st.subheader("Training History")
        
        max_len = 0
        for key in ['epochs', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']:
            if key in self.train_history and self.train_history[key] is not None:
                max_len = max(max_len, len(self.train_history[key]))
        
        # If no data found, use a default
        if max_len == 0:
            max_len = 1
        
        def pad_list(lst, target_len):
            if lst is None or len(lst) == 0:
                return [None] * target_len
            if len(lst) < target_len:
                return lst + [None] * (target_len - len(lst))
            return lst[:target_len]
        
        # Create DataFrame with same length on all columns
        history_df = pd.DataFrame({
            'Epoch': pad_list(
                self.train_history.get('epochs', list(range(1, max_len + 1))),
                max_len
            ),
            'Train Loss': pad_list(
                self.train_history.get('train_loss'),
                max_len
            ),
            'Val Loss': pad_list(
                self.train_history.get('val_loss'),
                max_len
            ),
            'Train Accuracy': pad_list(
                self.train_history.get('train_accuracy'),
                max_len
            ),
            'Val Accuracy': pad_list(
                self.train_history.get('val_accuracy'),
                max_len
            )
        })
        
        st.subheader("Loss")
        loss_df = history_df[['Epoch', 'Train Loss', 'Val Loss']].set_index('Epoch')

        loss_df = loss_df.dropna(how='all')
        if not loss_df.empty:
            st.line_chart(loss_df)
        else:
            st.info("No loss data available for plotting")
        
        # Plot Accuracy
        st.subheader("Accuracy")
        acc_df = history_df[['Epoch', 'Train Accuracy', 'Val Accuracy']].set_index('Epoch')
        # Remove None values for plotting
        acc_df = acc_df.dropna(how='all')
        if not acc_df.empty:
            st.line_chart(acc_df)
        else:
            st.info("No accuracy data available for plotting")
        
        with st.expander("History Table"):
            st.dataframe(history_df, use_container_width=True)
        
        # Metrics Section
        st.header("Metrics")
        
        # Create tabs for train/val/test
        tab1, tab2, tab3 = st.tabs(["Train", "Validation", "Test"])
        
        for tab, dataset in zip([tab1, tab2, tab3], ['train', 'val', 'test']):
            with tab:
                metrics_data = self.metrics['metrics'][dataset]
                
                st.subheader("General Metrics")
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
                
                st.subheader("Per-Class Metrics")
                num_classes = len(metrics_data['per_class_precision'])
                per_class_df = pd.DataFrame({
                    'Class': [f'Class {i}' for i in range(num_classes)],
                    'Precision': metrics_data['per_class_precision'],
                    'Recall': metrics_data['per_class_recall'],
                    'F1-Score': metrics_data['per_class_f1']
                })
                st.dataframe(per_class_df, use_container_width=True)
                
                st.subheader("Confusion Matrix")
                cm_df = pd.DataFrame(
                    metrics_data['confusion_matrix'],
                    columns=[f'Pred {i}' for i in range(num_classes)],
                    index=[f'True {i}' for i in range(num_classes)]
                )
                st.dataframe(cm_df, use_container_width=True)