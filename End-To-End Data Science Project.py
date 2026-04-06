# =========================================
# END-TO-END DATA SCIENCE PROJECT
# CUSTOMER CHURN PREDICTION
# =========================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle

# =========================
# 2. DATA COLLECTION
# =========================
print("Loading dataset...")

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Dataset Loaded Successfully!\n")

# =========================
# 3. DATA UNDERSTANDING
# =========================
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# 4. DATA PREPROCESSING
# =========================

# Drop unnecessary column
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values
df.dropna(inplace=True)

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

print("\nAfter preprocessing shape:", df.shape)

# =========================
# 5. FEATURE & TARGET
# =========================
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Convert boolean → int (fix for plotting)
y = y.astype(int)

# =========================
# 6. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. FEATURE SCALING
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 8. MODEL BUILDING
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# =========================
# 9. MODEL EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 10. VISUALIZATION
# =========================
plt.figure()
plt.hist(y, bins=2)
plt.title("Churn Distribution (0 = No, 1 = Yes)")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# =========================
# 11. SAVE MODEL
# =========================
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("\nModel and scaler saved successfully!")

# =========================
# 12. SAMPLE PREDICTION (FIXED)
# =========================
# ✅ Keep as DataFrame to avoid warning
sample = X.iloc[[0]]  

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nSample Prediction (0 = No Churn, 1 = Churn):", prediction[0])
