import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load trained model
model_file = "xgb_model.pkl"
try:
    with open(model_file, "rb") as file:
        best_xgb_model = pickle.load(file)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"Error loading model: {e}")
    best_xgb_model = None

# Load dataset for encoding categorical values
train_file = "cleaned_Vegetable_market.csv"
df = pd.read_csv(train_file)

# Prepare label encoders for categorical features
label_encoders = {}
X = df.drop(columns=["Price_per_kg"], errors='ignore')  # Keep only features
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if best_xgb_model is None:
            return render_template('index.html', prediction_text='Error: Model not found. Please retrain the model.')

        # Get user input from form
        data = {key: [float(value)] for key, value in request.form.items()}
        df_input = pd.DataFrame.from_dict(data)

        # Encode categorical values
        for col, le in label_encoders.items():
            if col in df_input.columns:
                df_input[col] = le.transform(df_input[col])

        # Make prediction
        prediction = best_xgb_model.predict(df_input)

        return render_template('index.html', prediction_text=f'Predicted Price: {prediction[0]:.2f} per kg')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
