import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Replace this with your actual CSV file name later
data = pd.read_csv("transactions.csv")

print("First 5 rows of dataset:")
print(data.head())

# -----------------------------
# 2. Basic Data Cleaning
# -----------------------------
# Drop duplicates if any
data = data.drop_duplicates()

# Handle missing values (simple approach)
data = data.dropna()

# -----------------------------
# 3. Exploratory Analysis
# -----------------------------
print("\nDataset Info:")
print(data.info())

print("\nFraud value counts:")
print(data["is_fraud"].value_counts())

# -----------------------------
# 4. Feature Selection
# -----------------------------
# Assume 'is_fraud' is target column
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Keep only numeric columns for simplicity
X = X.select_dtypes(include=[np.number])

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 7. Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 8. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_scaled)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 9. Prediction Function
# -----------------------------
def predict_fraud(transaction_features):
    """
    transaction_features: list or array of numeric features in the same order as X columns
    returns: probability of fraud and predicted class
    """
    transaction_features = np.array(transaction_features).reshape(1, -1)
    transaction_features_scaled = scaler.transform(transaction_features)

    prob = model.predict_proba(transaction_features_scaled)[0][1]
    prediction = model.predict(transaction_features_scaled)[0]

    return prob, prediction


# -----------------------------
# 10. Example Usage
# -----------------------------
# Example: replace with real values based on your dataset columns
# example_transaction = [1000, 2, 45, 1]  # dummy example
# prob, pred = predict_fraud(example_transaction)
# print("Fraud Probability:", prob)
# print("Prediction (1=Fraud, 0=Normal):", pred)
