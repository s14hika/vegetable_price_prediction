import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load dataset
train_file = "cleaned_Vegetable_market.csv"
df = pd.read_csv(train_file)

# Selecting relevant features
selected_features = ["Vegetable", "Season", "Month", "Temp"]
X = df[selected_features]
y = df["Price_per_kg"]

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
best_xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
best_xgb_model.fit(X_train, y_train)

# Save model
with open("xgb_model.pkl", "wb") as file:
    pickle.dump(best_xgb_model, file)

print("✅ Model trained and saved successfully as xgb_model.pkl!")
