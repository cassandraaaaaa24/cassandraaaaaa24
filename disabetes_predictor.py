import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("diabetes.csv")

# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model (Logistic Regression is simple and effective here)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Function to predict risk based on user input
def predict_diabetes():
    print("\nEnter your health values:")
    pregnancies = int(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = int(input("Age: "))

    # Create input array
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]])
    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)[0]
    if prediction == 1:
        print("\n⚠️ High risk of diabetes.")
    else:
        print("\n✅ Low risk of diabetes.")

# Run prediction
predict_diabetes()
