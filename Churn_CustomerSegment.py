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
st.title("Churn Rate by Customer Segments")

# Selection for analysis type
option = st.selectbox("Select a Customer Segment:", ["Product Category", "Customer Location"])

# Prepare data for visualization
churn_data = df.groupby([option, "Churn Risk"]).size().reset_index(name='Count')

# Plot the interactive clustered column chart
fig = px.bar(churn_data, x=option, y='Count', color='Churn Risk', 
             barmode='group', title=f"Churn Rate by {option}",
             labels={"Count": "Number of Customers"}, height=600)

# Display chart in Streamlit
st.plotly_chart(fig)