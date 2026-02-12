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

# Create sliders dynamically based on dataset min/max values
slider_values = {}
for column in X.columns:
    min_val = float(X[column].min())
    max_val = float(X[column].max())
    mean_val = float(X[column].mean())
    slider_values[column] = st.slider(column.replace("_", " ").title(), min_val, max_val, mean_val)

# Prediction Button
if st.button("Predict"):
    input_data = np.array([list(slider_values.values())])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🍷 Good Quality Wine")
    else:
        st.error("⚠️ Bad Quality Wine")
