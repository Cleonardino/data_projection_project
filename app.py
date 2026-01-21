import streamlit as st
import numpy as np
import pandas as pd
from src.summaries import load_experiments
from app.modeldataclass import ModelData


model_data: list[ModelData] = []

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

model_pages: list = []

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