import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
file_path = "ECommerce.csv"  # Ensure the file is in the same directory
@st.cache_data
def load_data():
    return pd.read_csv(file_path)

df = load_data()

# Streamlit App
st.title("Churn Rate by Purchase Amount & CLV (Interactive Box Plot)")

# Selection for Y-Axis
y_axis_option = st.selectbox("Select Y-Axis:", ["Customer Lifetime Value (USD)", "Purchase Amount (USD)"])

# Plotting Box Plot
fig = px.box(df, x="Churn Risk", y=y_axis_option, color="Churn Risk",
             title=f"Churn Risk vs {y_axis_option} (Box Plot)", height=600)

# Display chart in Streamlit
st.plotly_chart(fig)
