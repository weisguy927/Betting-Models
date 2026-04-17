# 📌 Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving/loading the model

# 📌 Load the Merged Dataset
file_path = "CBB Model (Ver 3).xlsx"  # Change to your file location
xls = pd.ExcelFile(file_path)

# Load historical betting data
historical_betting_data = pd.read_excel(xls, sheet_name="Historical Betting Data")

# Select relevant features for predictions
features = ["moneyline_home", "moneyline_away", "spread_home", "spread_away", "total_over", "total_under"]

# Drop missing values for selected features
historical_betting_data = historical_betting_data.dropna(subset=features)

# 📌 Define Features (X) and Target Variable (y)
X = historical_betting_data[features].fillna(0)  # Replace missing values with 0
y = (historical_betting_data["moneyline_home"] < historical_betting_data["moneyline_away"]).astype(int)  # Home win = 1, Away win = 0

# 📌 Split Data into Training and Test Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train a Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 📌 Make Predictions on the Test Set
y_pred = rf_model.predict(X_test)

# 📌 Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Random Forest Accuracy: {accuracy:.4f}")

# 📌 Save the Trained Model (Optional)
joblib.dump(rf_model, "betting_model.pkl")
print("✅ Model saved as 'betting_model.pkl' for future use.")

# 📌 To Load and Use the Model Later:
# rf_model = joblib.load("betting_model.pkl")
# predictions = rf_model.predict(X_test)
