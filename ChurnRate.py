import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv("ecommerce_customer.csv")  

# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Change the format of the date to "YYYY-MM"
df['Order Month'] = df['Order Date'].dt.strftime('%Y-%m')

'''Exploratory Data Analysis (EDA)'''
# Checking for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Checking the distribution of Churn Risk
churn_risk_distribution = df['Churn Risk'].value_counts(normalize=True) * 100
print(churn_risk_distribution)

