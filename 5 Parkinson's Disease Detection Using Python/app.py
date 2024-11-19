import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv(r"C:\Users\vikas\100-Project-Data-science\5 Parkinson's Disease Detection Using Python\dataset.csv")

# Preprocessing the data
# Drop the 'name' column and the 'status' column as it is the target
X = df.drop(columns=['name', 'status'], axis=1)  # Features
y = df['status']  # Target variable

# Feature names excluding 'name' and 'status'
FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
    'spread2', 'D2', 'PPE'
]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Streamlit Input form
st.title("Parkinson's Disease Detection")
st.write("Enter the values for the following features:")


input_data = []
for feature in FEATURE_NAMES:
    value = st.number_input(f"{feature}", min_value=0.0, max_value=100.0, step=0.01)
    input_data.append(value)

# Button to make the prediction
if st.button('Predict Parkinson\'s Disease'):
    # Convert the input data to a numpy array and reshape for prediction
    input_array = np.asarray(input_data).reshape(1, -1)

    # Debugging: Print the shape of the input data
    st.write(f"Input shape: {input_array.shape}")

    if input_array.shape[1] == len(FEATURE_NAMES):  
        # Standardize the input data
        input_scaled = scaler.transform(input_array)
        
        # Prediction
        prediction = model.predict(input_scaled)
        
        # Show result
        if prediction[0] == 0:
            st.write("**Result: Healthy (No Parkinson's Disease)**")
        else:
            st.write("**Result: Parkinson's Disease**")
    else:
        st.error("Input data does not match the required number of features.")
