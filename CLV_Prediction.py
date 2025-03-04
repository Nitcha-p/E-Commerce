import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Streamlit App Title ---
st.title("📊 CLV Prediction & Forecasting with Prophet")
st.markdown("This app forecasts Customer Lifetime Value (CLV) using Prophet and automatically selects the best Moving Average window.")

# --- Load Your Own CSV ---
df = pd.read_csv("CLV_TrendOverTime.csv")  # YOUR CSV FILE

# Ensure Date column is in datetime format
df['Order Month'] = pd.to_datetime(df['Order Month'])

# Fill missing values (Backfill + Forward Fill)
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

# --- User Input for Forecasting Period ---
forecast_period = st.number_input("Enter the number of months to predict :", min_value=1, max_value=36, value=6)

# --- Define Moving Average Windows ---
ma_options = [1, 3, 5, 7]  # MA Windows
results = {}

# --- Function to Prepare Data ---
def prepare_data(df, column):
    df_prophet = df[['Order Month', column]].rename(columns={'Order Month': 'ds', column: 'y'})
    return df_prophet.dropna()

# --- Prophet Model Training & Forecasting ---
def evaluate_prophet(df, column, forecast_period):
    data = prepare_data(df, column)
    model = Prophet()
    model.fit(data)
    
    # Forecast future values
    future = model.make_future_dataframe(periods=forecast_period, freq='M')
    forecast = model.predict(future)

    # Ensure we are evaluating predictions on future data only
    y_true = df[column].iloc[-forecast_period:].values  # Last n months from dataset
    y_pred = forecast['yhat'].iloc[-forecast_period:].values  # Future predictions

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return forecast, mae, rmse

# --- Evaluate All Moving Averages ---
for ma in ma_options:
    df[f'CLV_MA_{ma}M'] = df['CLV (Monthly)'].rolling(window=ma, min_periods=1).mean()
    forecast, mae, rmse = evaluate_prophet(df, f'CLV_MA_{ma}M', forecast_period)
    results[ma] = {"Forecast": forecast, "MAE": mae, "RMSE": rmse}

# --- Select the Best Model ---
best_ma = min(results, key=lambda x: results[x]["RMSE"])
best_forecast = results[best_ma]["Forecast"]
best_mae = results[best_ma]["MAE"]
best_rmse = results[best_ma]["RMSE"]

# --- Display Results ---
st.subheader("📌 Forecasting Results")
st.write(f"✅ **Best Moving Average Model :** {best_ma} Months")
st.write(f"📉 **MAE (Mean Absolute Error) :** {best_mae:.2f}")
st.write(f"📊 **RMSE (Root Mean Squared Error) :** {best_rmse:.2f}")

# --- Merge Actual & Predicted Data ---
best_forecast = best_forecast[['ds', 'yhat']].copy()
df_actual = df[['Order Month', f'CLV_MA_{best_ma}M']].rename(columns={'Order Month': 'ds', f'CLV_MA_{best_ma}M': 'Actual CLV'})
df_merged = df_actual.merge(best_forecast, on='ds', how='outer').rename(columns={'yhat': 'Predicted CLV'})

# --- Plot CLV Forecast ---
st.subheader("📈 CLV Forecasting Visualization")
fig = px.line(df_merged, x="ds", y=["Actual CLV", "Predicted CLV"], 
              labels={"ds": "Month", "value": "CLV (USD)"},
              title=f"<b>Projected CLV with Prophet (Next {forecast_period} Months, Best MA {best_ma}M)</b>")
st.plotly_chart(fig, use_container_width=True)
