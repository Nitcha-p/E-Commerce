from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px

# Load the dataset
df = pd.read_csv("CLV_TrendOverTime.csv")

# Check missing values
print(df.isnull().sum())

# Fill initial missing values for each column using backfill
df.fillna(method='bfill',inplace=True)
df.fillna(method='ffill',inplace=True)
print(df)

# เตรียมข้อมูลสำหรับ Prophet
def prepare_data(df, column):
    df_prophet = df[['Order Month', column]].rename(columns={'Order Month': 'ds', column: 'y'})
    return df_prophet.dropna()

# ฟังก์ชันทดสอบโมเดล Prophet และคำนวณ Accuracy
def evaluate_prophet(df, column):
    data = prepare_data(df, column)

    # สร้างโมเดล Prophet
    model = Prophet()
    model.fit(data)

    # พยากรณ์ 6 เดือนข้างหน้า
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)

    # คำนวณ MAE และ RMSE
    y_true = data['y']
    y_pred = forecast['yhat'][:len(y_true)]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return forecast, mae, rmse

# ทดลองกับทุกตัวแปร
columns_to_test = ["CLV (Monthly)", "CLV Moving Average (3M)", "CLV Moving Average (5M)", "CLV Moving Average (7M)"]
results = {}

for col in columns_to_test:
    forecast, mae, rmse = evaluate_prophet(df, col)
    results[col] = {"Forecast": forecast, "MAE": mae, "RMSE": rmse}

# แสดงผลลัพธ์
best_model = min(results, key=lambda x: results[x]["RMSE"])
print(f"Best Model : {best_model}")
print(f"MAE : {results[best_model]['MAE']}")
print(f"RMSE : {results[best_model]['RMSE']}")

# Extract the best forecasted data
best_forecast = results[best_model]["Forecast"][['ds', 'yhat']]
best_forecast = best_forecast.copy()
best_forecast.loc[:, 'ds'] = pd.to_datetime(best_forecast['ds'])

# Extract actual CLV from the dataset
df_actual = df[['Order Month', best_model]].rename(columns={'Order Month': 'ds', best_model: 'Actual CLV'})
df_actual['ds'] = pd.to_datetime(df_actual['ds'])

# Merge actual and predicted data
df_merged = df_actual.merge(best_forecast, on='ds', how='outer')
df_merged = df_merged.rename(columns={'yhat': 'Predicted CLV'})

# Create an interactive chart with Actual CLV and Predicted CLV (yhat)
fig = px.line(df_merged, x="ds", y=["Actual CLV", "Predicted CLV"],
              labels={"ds": "Month", "value": "CLV (USD)"},
              title="<b>Projected Customer Lifetime Value (6-Month Forecast, MA 7M)</b>")

# Center the title
fig.update_layout(title_x=0.5)

fig.show()