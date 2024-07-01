import os
from flask import Flask, request, redirect, url_for, send_file, render_template, flash
import pandas as pd
from werkzeug.utils import secure_filename
import logging

# Import functions from your existing model script
from untitled import load_data, preprocess_data, train_model, evaluate_model, predict_new_data, save_predictions

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predictions_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')

            # Log the file paths
            logging.debug(f"File uploaded to: {file_path}")
            logging.debug(f"Predictions file will be saved to: {predictions_file}")

            try:
                # Load training and test data
                train_file = 'train.csv'  # Update this path as necessary
                train_data = load_data(train_file)
                test_data = load_data(file_path)

                if train_data is None or test_data is None:
                    flash('Error loading data files.')
                    return redirect(request.url)

                # Preprocess data (drop unnecessary columns and encode categorical variables)
                y = train_data['SalePrice']
                X = train_data.drop(columns=['SalePrice'])
                X_test = test_data

                # Preprocess combined data to ensure consistent column structure
                X, X_test = preprocess_data(X, X_test)

                # Handle missing values with SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

                # Scale the features
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

                # Split the data into training and validation sets
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                model = train_model(X_train, y_train, 42)

                # Evaluate the model
                evaluate_model(model, X_val, y_val)

                # Make predictions on test data
                predictions = predict_new_data(model, X_test, X.columns)

                # Save predictions to a CSV file
                save_predictions(predictions, test_data, predictions_file)

                # Log success
                logging.debug("Predictions saved successfully.")
                return send_file(predictions_file, as_attachment=True)

            except Exception as e:
                logging.error(f"Error during processing: {e}")
                flash('An error occurred during processing.')
                return redirect(request.url)

        flash('Invalid file type')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
