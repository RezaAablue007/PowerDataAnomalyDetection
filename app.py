from flask import Flask, render_template, request, send_file
from analysis import load_data, stl_anomaly_detection, linear_regression_forecast, random_forest_forecast, plot_forecast, plot_residuals, calculate_accuracy_metrics, get_image_data
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
Bootstrap(app)

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv', 'txt'}

def analyze_data(file_path):
    data = load_data(file_path)

    if isinstance(data, str):
        return f"Error loading data: {data}", None

    # Initialize dictionaries for metrics
    lr_metrics_dict = {}
    rf_metrics_dict = {}

    results = {}

    for column in ['Market Demand', 'Ontario Demand']:
        anomaly_indices, residuals, seasonal_anomalies_dict, monthly_anomalies_dict = stl_anomaly_detection(data, column)
        lr_forecast = linear_regression_forecast(data, column)
        rf_forecast = random_forest_forecast(data, column)

        # Calculate metrics for both models
        lr_metrics = calculate_accuracy_metrics(data[column][:len(lr_forecast)], lr_forecast)
        rf_metrics = calculate_accuracy_metrics(data[column][:len(rf_forecast)], rf_forecast)

        # Store metrics in dictionaries
        lr_metrics_dict[column] = lr_metrics
        rf_metrics_dict[column] = rf_metrics

        # Plotting and handling figure objects
        lr_plot = plot_forecast(data, lr_forecast, "Linear Regression Forecast", column)
        lr_residual_plot = plot_residuals(data, lr_forecast, "Linear Regression", column)

        rf_plot = plot_forecast(data, rf_forecast, "Random Forest Forecast", column)
        rf_residual_plot = plot_residuals(data, rf_forecast, "Random Forest", column)

        ad_plot = plot_forecast(data, None, "Anomaly Detection", column, anomaly_indices)

        results[column] = {
            'lr_metrics': lr_metrics,
            'rf_metrics': rf_metrics,
            'lr_plot': get_image_data(lr_plot),
            'lr_residual_plot': get_image_data(lr_residual_plot),
            'rf_plot': get_image_data(rf_plot),
            'rf_residual_plot': get_image_data(rf_residual_plot),
            'ad_plot': get_image_data(ad_plot),
            'seasonal_anomalies_dict': seasonal_anomalies_dict,
        }

    return None, results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'FILE_PATH' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['FILE_PATH']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Securely save the uploaded file to a designated uploads folder
        uploads_folder = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join(uploads_folder, filename)
        file.save(file_path)

        # Now 'file_path' contains the path to the uploaded file

        error, results = analyze_data(file_path)

        if error:
            return render_template('index.html', error=error)

        return render_template('results.html', results=results)

    return render_template('index.html')

@app.route('/download_python_file', methods=['GET'])
def download_python_file():
    python_file_path = os.path.join(os.getcwd(), 'AnomalyDetectionSoftware.py')
    return send_file(python_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False)
