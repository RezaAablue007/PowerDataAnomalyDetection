import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pyod.models.knn import KNN
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64

# Turn off interactive mode in matplotlib
plt.ioff()

# Classify the date into a season
def classify_season(date):
    month = date.month
    if month in [4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

# Perform STL decomposition and detect anomalies
def stl_anomaly_detection(data, column, period=24):
    try:
        stl = STL(data[column], period=period, robust=True)
        result = stl.fit()
        residuals = result.resid
        knn = KNN(contamination=0.01)
        knn.fit(residuals.values.reshape(-1, 1))
        anomalies = knn.predict(residuals.values.reshape(-1, 1))
        data['Season'] = data['DateTime'].apply(lambda x: classify_season(x))
        data['Month'] = data['DateTime'].dt.strftime('%B')  # Add month names
        anomaly_indices = np.where(anomalies == 1)[0]
        seasonal_anomalies = data.iloc[anomaly_indices].groupby('Season')[column].count()
        monthly_anomalies = data.iloc[anomaly_indices].groupby('Month')[column].count()  # Group by month names
        seasonal_anomalies_dict = seasonal_anomalies.to_dict()
        monthly_anomalies_dict = monthly_anomalies.to_dict()  # New dictionary with month names
        return anomaly_indices, residuals, seasonal_anomalies_dict, monthly_anomalies_dict
    except Exception as e:
        raise RuntimeError(f"Error in STL anomaly detection: {e}")

# Preprocess data for better forecasting
def preprocess_data(data, column):
    data['Time'] = np.arange(len(data))
    data['Sin_Time'] = np.sin(2 * np.pi * data['Time'] / 8760)
    data['Cos_Time'] = np.cos(2 * np.pi * data['Time'] / 8760)
    return data

# Forecast using Linear Regression model for the next year
def linear_regression_forecast(data, column):
    preprocessed_data = preprocess_data(data, column)
    features = ['Sin_Time', 'Cos_Time']
    X = preprocessed_data[features]
    y = preprocessed_data[column].values
    model = LinearRegression()
    model.fit(X, y)
    forecast_time = np.arange(len(preprocessed_data), len(preprocessed_data) + 8760)
    forecast_X = pd.DataFrame({
        'Sin_Time': np.sin(2 * np.pi * forecast_time / 8760),
        'Cos_Time': np.cos(2 * np.pi * forecast_time / 8760),
    })
    forecast = model.predict(forecast_X)
    return forecast

# Forecast using Random Forest model for the next year
def random_forest_forecast(data, column):
    preprocessed_data = preprocess_data(data, column)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    X = preprocessed_data[[column]].shift(1).fillna(method='bfill')
    y = preprocessed_data[column]
    rf_model.fit(X, y)
    forecast_X = X.tail(8760).values.reshape(-1, 1)
    forecast = rf_model.predict(forecast_X)
    return forecast

# Plot forecast with anomalies - now returns a figure object
def plot_forecast(data, forecast, title, column, anomalies=None, residuals=None):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.plot(data['DateTime'], data[column], label='Original Data')
    if forecast is not None:
        forecast_dates = pd.date_range(start=data['DateTime'].iloc[-1] + pd.Timedelta(hours=1), periods=len(forecast), freq='H')
        ax.plot(forecast_dates, forecast, label='Forecasted Data', color='orange')
    if anomalies is not None and residuals is None:
        anomaly_dates = data['DateTime'].iloc[anomalies]
        ax.scatter(anomaly_dates, data[column].iloc[anomalies], color='red', label='Anomalies', zorder=5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.set_title(f"{title} - {column}")
    ax.set_xlabel('Date (YYYY-MM)')
    ax.set_ylabel('Power Demand (MW)')
    ax.legend()
    plt.tight_layout()
    return fig  # Return the figure object

# Calculate accuracy metrics for the forecasted data
def calculate_accuracy_metrics(original_data, forecasted_data):
    mape = np.mean(np.abs((original_data - forecasted_data) / original_data)) * 100
    mad = np.mean(np.abs(original_data - forecasted_data))
    mse = mean_squared_error(original_data, forecasted_data)
    rmse = np.sqrt(mse)
    r2 = r2_score(original_data, forecasted_data)
    mae = mean_absolute_error(original_data, forecasted_data)
    return mape, mad, mse, rmse, r2, mae

# Plot residuals for the forecasting method - now returns a figure object
def plot_residuals(data, forecasted_data, title, column):
    residuals = data[column] - forecasted_data[:len(data)]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.plot(data['DateTime'], residuals, label='Residuals', color='blue')
    ax.set_title(f"{title} - Residuals")
    ax.set_xlabel('Date (YYYY-MM)')
    ax.set_ylabel('Residual Value')
    ax.legend()
    plt.tight_layout()
    return fig  # Return the figure object

# Function to convert plots to a format suitable for PySimpleGUI
def get_image_data(plt_figure, **kwargs):
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png', **kwargs)
    buf.seek(0)
    # img_data = buf.getvalue()  # Return image data as a byte string
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    return img_data

# Load data from the CSV file
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data['DateTime'] = data['Date'] + pd.to_timedelta(data['Hour'] - 1, unit='h')
        return data
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
    