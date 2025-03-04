'''CLV Trend Over Time (Monthly)'''
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv("ecommerce_customer.csv")  

# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Change the format of the date to "YYYY-MM"
df['Order Month'] = df['Order Date'].dt.strftime('%Y-%m')
print(df.head())

# Calculate the average CLV per month
CLV = df.groupby('Order Month')['Customer Lifetime Value (USD)'].mean().reset_index()
CLV = CLV.rename(columns={'Customer Lifetime Value (USD)': 'CLV (Monthly)'})


'''CLV Trend Over Time using Moving Average (3 Months and 7 Months)'''
# Calcualte Moving Average 3 months and 7 months
CLV['CLV Moving Average (3M)'] = CLV['CLV (Monthly)'].rolling(window=3, center=True).mean()
CLV['CLV Moving Average (5M)'] = CLV['CLV (Monthly)'].rolling(window=5, center=True).mean()
CLV['CLV Moving Average (7M)'] = CLV['CLV (Monthly)'].rolling(window=7, center=True).mean()

# Create an interactive line chart for CLV trend (3 Months)
fig = px.line(CLV, x="Order Month", y=["CLV (Monthly)", "CLV Moving Average (3M)",'CLV Moving Average (5M)','CLV Moving Average (7M)'],
              labels={"value": "CLV (USD)", "Order Month": "Month"},
              title="<b>CLV Trend Over Time</b>",
              markers=True)

# Center the title
fig.update_layout(title_x=0.5)

fig.show()

# Save CLV Dataframe to CSV
CLV.to_csv("CLV_TrendOverTime.csv", index=False)

