import PySimpleGUI as sg
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
import webbrowser

# Turn off interactive mode in matplotlib
plt.ioff()

# Show descriptions window
def show_descriptions():
    descriptions_text = """
Machine Learning Overview

What is Machine Learning?
Machine learning involves algorithms that enable computer programs to learn and make decisions or predictions from data. 
It's a process where a program can identify patterns and adjust its actions accordingly without being explicitly programmed for every possible scenario.

Forecasting Methods:

1. Linear Regression Forecasting

What is it? 
- A technique to predict the linear relationship between input and output variables.
Pros: Simple, straightforward, and efficient.
Cons: Assumes a linear relationship and may oversimplify data.

2. Random Forest Forecasting

What is it? 
- An ensemble method using multiple decision trees to improve prediction accuracy.
Pros: High accuracy and ability to model nonlinear relationships.
Cons: Can be biased with biased training data and requires significant memory.

Anomaly Detection in Power Meters:

Definition: Identifying unusual patterns or changes in power consumption that deviate from the norm, aimed at spotting equipment malfunctions or inefficiencies.

KNN Algorithm for Anomaly Detection:

How it Works: Utilizes the principle of nearest neighbors to detect anomalies by comparing data points to their closest counterparts.

Understanding Forecasting Accuracy Metrics:

MAPE (Mean Absolute Percentage Error): Measures the average percentage error between predicted and actual values.
MAE (Mean Absolute Error): Calculates the average of the absolute errors between predicted and actual values.
MSE (Mean Squared Error): Computes the average of the squared differences between predicted and actual values.
RMSE (Root Mean Squared Error): Provides error measurement in the same units as the data by taking the square root of MSE.
MAD (Mean Absolute Deviation): Finds the average absolute difference between each data point and the overall mean.
R2-Squared (R-squared): Indicates the percentage of the variance in the dependent variable that is predictable from the independent variables.

Residual Curves Explained:

Purpose: Residual curves plot the differences between actual and predicted values, offering insight into a forecasting model's accuracy.
    """
    layout = [
        [sg.Text("Overview", font=("Helvetica", 16))],
        [sg.Multiline(descriptions_text, size=(80, 20), font=("Helvetica", 12), disabled=True)],
        [sg.Button("Close")]
    ]
    window = sg.Window("Overview", layout, modal=True)
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Close"):
            break
    window.close()

# Load data from the CSV file
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data['DateTime'] = data['Date'] + pd.to_timedelta(data['Hour'] - 1, unit='h')
        return data
    except Exception as e:
        return str(e)

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
        print(f"Error in STL anomaly detection: {e}")
        return [], pd.Series(dtype='float64'), {}, {}

# Preprocess data for better forecasting
def preprocess_data(data, column):
    data['Time'] = np.arange(len(data))
    data['Sin_Time'] = np.sin(2 * np.pi * data['Time'] / 8760)
    data['Cos_Time'] = np.cos(2 * np.pi * data['Time'] / 8760)
    return data

# Forecast using Linear Regression model for the next year
def linear_regression_forecast(data, column, forecast_periods):
    preprocessed_data = preprocess_data(data, column)
    features = ['Sin_Time', 'Cos_Time']
    X = preprocessed_data[features]
    y = preprocessed_data[column].values
    model = LinearRegression()
    model.fit(X, y)
    forecast_time = np.arange(len(data))
    forecast_X = pd.DataFrame({
        'Sin_Time': np.sin(2 * np.pi * forecast_time / 8760),
        'Cos_Time': np.cos(2 * np.pi * forecast_time / 8760),
    })
    forecast = model.predict(forecast_X)
    return forecast[:forecast_periods]

# Forecast using Random Forest model for the next year
def random_forest_forecast(data, column, forecast_periods):
    preprocessed_data = preprocess_data(data, column)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    X = preprocessed_data[[column]].shift(1).fillna(method='bfill')
    y = preprocessed_data[column]
    rf_model.fit(X, y)
    forecast_X = X.tail(len(data)).values.reshape(-1, 1)
    forecast = rf_model.predict(forecast_X)
    return forecast[:forecast_periods]

# Plot forecast with anomalies - now returns a figure object
def plot_forecast(data, forecast, title, column, anomalies=None, residuals=None):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.plot(data['DateTime'], data[column], label='Original Data', color='blue')
    if forecast is not None:
        # Align forecast_dates with the original data's DateTime
        forecast_dates = data['DateTime']
        ax.plot(forecast_dates, forecast, label='Forecasted Data', color='orange', linestyle='--')
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
    img_data = buf.getvalue()  # Return image data as a byte string
    print("Type of image data:", type(img_data))  # Debug print
    print("Length of image data:", len(img_data))  # Debug print
    return img_data

def main():
    while True:
        layout = [
            [sg.Text("Smart Meter Data Analysis Tool (BV04 Capstone Project)", font=("Helvetica", 16))],
            [sg.Text("Made by: Reza Aablue, Danial Nadeem, Aavash Neupane, and Rahman Saeed", font=("Helvetica", 12))],
            [sg.Text("Welcome! Please enter the path of the .csv file to be analyzed:")],
            [sg.InputText(key='FILE_PATH'), sg.FileBrowse()],
            [sg.Submit(), sg.Cancel()],
            [sg.Button("Overview about the software's outputs", key="SHOW_DESC")],
            [sg.Text("Access the Web Application version of our software (click here)", tooltip="http://example.com", enable_events=True, key="CLOUD_LINK")]
        ]

        window = sg.Window('File Input', layout)

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, 'Cancel'):
                break
            if event == "SHOW_DESC":
                show_descriptions()  # This will show the descriptions window
            

            if event == 'Submit':
                file_path = values['FILE_PATH']
                window.close()
                data = load_data(file_path)

                if isinstance(data, str):
                    sg.popup_error(f"Error loading data: {data}")
                    continue

                # Initialize dictionaries for metrics
                lr_metrics_dict = {}
                rf_metrics_dict = {}

                forecast_periods = len(data)  # Use the length of the original data for the forecast period

                for column in ['Market Demand', 'Ontario Demand']:
                    anomaly_indices, residuals, seasonal_anomalies_dict, monthly_anomalies_dict = stl_anomaly_detection(data, column)
                    lr_forecast = linear_regression_forecast(data, column, forecast_periods)
                    rf_forecast = random_forest_forecast(data, column, forecast_periods)

                    # Calculate metrics for both models
                    lr_metrics = calculate_accuracy_metrics(data[column][:len(lr_forecast)], lr_forecast)
                    rf_metrics = calculate_accuracy_metrics(data[column][:len(rf_forecast)], rf_forecast)

                    # Store metrics in dictionaries
                    lr_metrics_dict[column] = lr_metrics
                    rf_metrics_dict[column] = rf_metrics

                    # Plotting and handling figure objects
                    lr_plot = plot_forecast(data, lr_forecast, "Linear Regression Forecast", column)
                    lr_plot_data = get_image_data(lr_plot)  # This now returns a byte string
                    lr_plot.clf()  # Clear the figure to free up memory

                    rf_plot = plot_forecast(data, rf_forecast, "Random Forest Forecast", column)
                    rf_plot_data = get_image_data(rf_plot)  # This now returns a byte string
                    rf_plot.clf()

                    ad_plot = plot_forecast(data, None, "Anomaly Detection", column, anomaly_indices)
                    ad_plot_data = get_image_data(ad_plot)  # This now returns a byte string
                    ad_plot.clf()

                    lr_residual_plot = plot_residuals(data, lr_forecast, "Linear Regression", column)
                    lr_residual_plot_data = get_image_data(lr_residual_plot)  # This now returns a byte string
                    lr_residual_plot.clf()

                    rf_residual_plot = plot_residuals(data, rf_forecast, "Random Forest", column)
                    rf_residual_plot_data = get_image_data(rf_residual_plot)  # This now returns a byte string
                    rf_residual_plot.clf()

                    results_column = sg.Column([
                        [sg.Image(data=lr_plot_data)],
                        [sg.Image(data=lr_residual_plot_data)],
                        [sg.Text(f"Linear Regression MAPE: {lr_metrics[0]:.2f}%")],
                        [sg.Text(f"Linear Regression MAD: {lr_metrics[1]:.2f}")],
                        [sg.Text(f"Linear Regression MSE: {lr_metrics[2]:.2f}")],
                        [sg.Text(f"Linear Regression RMSE: {lr_metrics[3]:.2f}")],
                        [sg.Text(f"Linear Regression R2: {lr_metrics[4]:.2f}")],
                        [sg.Text(f"Linear Regression MAE: {lr_metrics[5]:.2f}")],
                        [sg.Image(data=rf_plot_data)],
                        [sg.Image(data=rf_residual_plot_data)],
                        [sg.Text(f"Random Forest MAPE: {rf_metrics[0]:.2f}%")],
                        [sg.Text(f"Random Forest MAD: {rf_metrics[1]:.2f}")],
                        [sg.Text(f"Random Forest MSE: {rf_metrics[2]:.2f}")],
                        [sg.Text(f"Random Forest RMSE: {rf_metrics[3]:.2f}")],
                        [sg.Text(f"Random Forest R^2: {rf_metrics[4]:.2f}")],
                        [sg.Text(f"Random Forest MAE: {rf_metrics[5]:.2f}")],
                        [sg.Image(data=ad_plot_data)],
                        [sg.Text(f"Number of anomalies detected in {column} by season:")],
                        [sg.Text(f"Spring - {seasonal_anomalies_dict.get('Spring', 0)}")],
                        [sg.Text(f"Summer - {seasonal_anomalies_dict.get('Summer', 0)}")],
                        [sg.Text(f"Fall - {seasonal_anomalies_dict.get('Fall', 0)}")],
                        [sg.Text(f"Winter - {seasonal_anomalies_dict.get('Winter', 0)}")],
                    ], scrollable=True, vertical_scroll_only=True, size=(1000, 1000))

                    result_layout = [
                        [sg.Text(f"Results for {column}")],
                        [results_column],
                        [sg.Button("Close")]
                    ]

                    result_window = sg.Window(f"Analysis Results - {column}", result_layout, resizable=True, finalize=True)
                    result_window.bring_to_front()

                    event, values = result_window.read()
                    if event in (sg.WIN_CLOSED, 'Close'):
                        result_window.close()
                break
            if event == "CLOUD_LINK":
                webbrowser.open("http://18.216.107.70")

        window.close()

        restart = sg.popup_yes_no("Do you want to analyze another file?")
        if restart == 'No':
            break
                
if __name__ == "__main__":
    main()