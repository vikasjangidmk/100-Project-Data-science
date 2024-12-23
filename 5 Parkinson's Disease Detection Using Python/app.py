import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv(r"C:\Users\vikas\100-Project-Data-science\5 Parkinson's Disease Detection Using Python\dataset.csv")

# Preprocessing the data
# Drop the 'name' column and 'status' column
X = df.drop(columns=['name', 'status'], axis=1)  # Features
y = df['status']  # Target variable

# Feature names
FEATURE_NAMES = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Streamlit App
st.title("Parkinson's Disease Detection")
st.write("Enter the values for the following features to predict Parkinson's Disease:")

# Collect inputs dynamically based on feature ranges
input_data = []
for feature in FEATURE_NAMES:
    min_val, max_val = X[feature].min(), X[feature].max()
    value = st.number_input(
        f"{feature} (Range: {min_val:.2f} to {max_val:.2f})",
        min_value=min_val,
        max_value=max_val,
        value=min_val,  # Default to the minimum value
        step=0.01
    )
    input_data.append(value)

# Button to make the prediction
if st.button('Predict Parkinson\'s Disease'):
    st.write("Input Data:", input_data)  # Debugging input data
    if all(value != 0.0 for value in input_data):  # Ensure all inputs are valid
        # Convert to NumPy array and reshape for prediction
        input_array = np.asarray(input_data).reshape(1, -1)

        # Standardize the input data
        input_scaled = scaler.transform(input_array)

        # Prediction
        prediction = model.predict(input_scaled)

        # Show result
        if prediction[0] == 0:
            st.write("*Result: Healthy (No Parkinson's Disease)*")
        else:
            st.write("*Result: Parkinson's Disease*")
    else:
        st.error("Please provide valid values for all features!")