import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🍷 Wine Quality Prediction")
st.write("Predict whether the wine is Good or Bad Quality")

# Load Dataset
wine_dataset = pd.read_csv("wine.csv")

# Data Preprocessing
X = wine_dataset.drop("quality", axis=1)
Y = wine_dataset["quality"].apply(lambda y_value: 1 if y_value >= 7 else 0)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

st.header("Enter Wine Details")

# Input fields using sliders
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.0)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.07)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 75.0, 15.0)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 300.0, 46.0)
density = st.slider("Density", 0.9900, 1.0050, 0.9960)
pH = st.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6)
alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)

# Prediction Button
if st.button("Predict"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🍷 Good Quality Wine")
    else:
        st.error("⚠️ Bad Quality Wine")
