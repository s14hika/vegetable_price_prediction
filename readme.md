# Vegetable Price Prediction

## Overview
The **Vegetable Price Prediction** project is a web-based application that predicts vegetable prices based on various input parameters such as vegetable type, season, month, temperature, disaster conditions, and market conditions. The application utilizes a trained **XGBoost model** to make predictions and is deployed using **Flask** as a web server.

## Features
- Predicts vegetable prices based on input parameters
- Uses **XGBoost model** for price prediction
- Web-based interface for easy user interaction
- Supports categorical encoding for vegetable types and seasons
- Handles missing or incorrect input data

## Technologies Used
- **Python** (Flask, Pandas, NumPy, XGBoost, Scikit-learn, Pickle)
- **Flask** (Web framework for deployment)
- **HTML, CSS** (Frontend interface)
- **VS Code** (Development environment)

## Project Structure
```
vegetable_price_prediction/
│-- templates/             # HTML templates for the web app
│   ├── index.html         # Main user interface
│-- cleaned_Vegetable_market.csv  # Preprocessed dataset
│-- xgb_model.pkl          # Trained XGBoost model
│-- app.py                 # Main Flask application
│-- train.py               # Model training script
│-- test.csv               # Test dataset
│-- README.md              # Project documentation
```

## Installation & Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/vegetable-price-prediction.git
   cd vegetable-price-prediction
   ```

2. **Install Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask Application:**
   ```bash
   python app.py
   ```
   The application will be available at: `http://127.0.0.1:5000/`

## Usage
1. Open the web application in a browser.
2. Enter the required input fields (Vegetable, Season, Month, Temperature, etc.).
3. Click on the "Predict Price" button.
4. View the predicted price displayed on the screen.

## Model Training
- The `train.py` script is used to preprocess the dataset and train an **XGBoost** model.
- The trained model is saved as `xgb_model.pkl` and is used for predictions in the web app.

## Troubleshooting
- If the Flask server does not start, ensure that all dependencies are installed correctly.
- If you receive a feature mismatch error, ensure that input fields match the model's expected format.
- For debugging, check the terminal logs when running `app.py`.

## Future Enhancements
- Improve the UI with better styling and responsiveness.
- Integrate a database for real-time price updates.
- Deploy the model using **Docker** or a cloud-based service.

## Contributors
- **Shaik Sadhika** - [GitHub Profile](https://github.com/s14hika)



