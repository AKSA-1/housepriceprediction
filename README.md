# housepriceprediction

# House Price Prediction with Flask and HistGradientBoostingRegressor

This project aims to predict house prices using a machine learning model, specifically the `HistGradientBoostingRegressor`, and a Flask web application for file uploads and predictions.

## Project Structure
├── uploads/ # Directory to store uploaded files
├── model.py # Python script containing the ML model and preprocessing functions
├── app.py # Flask application script
├── index.html # HTML template for the web interface
├── train.csv # Training dataset
├── README.md # This README file


## Setup and Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/house-price-prediction.git
    cd house-price-prediction
    ```

2. Create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure the `uploads` directory exists:
    ```sh
    mkdir uploads
    ```

## Running the Application

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. Open your web browser and go to:
    ```
    http://127.0.0.1:5000/
    ```

## Usage

1. Upload CSV for Prediction:
    - On the home page, use the form to upload a CSV file containing the test data.
    - The application will process the file, make predictions, and provide a download link for the predicted prices.

## File Descriptions

- `model.py`:
  - Contains functions for loading data, preprocessing, training the model, evaluating performance, and making predictions.
  
- `app.py`:
  - The Flask application that handles file uploads, data processing, and serves the HTML template.
  
- `index.html`:
  - The HTML template for the web interface where users can upload their CSV files for prediction.

## Example

1. Training Data (`train.csv`):
    - Ensure you have a training dataset in the root directory. This dataset should include features and a target column (`SalePrice`).

2. Prediction:
    - Upload a CSV file (without the `SalePrice` column) through the web interface.
    - The application will preprocess the data, make predictions using the trained model, and provide a CSV file with predicted prices.

## Dependencies

- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- werkzeug

Install these dependencies using the provided `requirements.txt` file:
```sh
pip install -r requirements.txt

 This README provides all necessary information for setting up, running, and troubleshooting the project, formatted for GitHub.
