import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Inject custom CSS styles
st.markdown("""
<style>
    /* Background color */
    .stApp {
        background-color: #d0e7f9;
        color: black;  /* Set all text to black */
    }

    /* Title style */
    h1 {
        color: black;
        font-size: 48px;
        font-weight: bold;
    }

    /* Bigger input boxes */
    div.stNumberInput > label > div {
        font-size: 18px;
        color: black;
    }
    div.stNumberInput > div > input {
        height: 45px;
        font-size: 20px;
        color: black;
    }

    /* Style the button */
    div.stButton > button:first-child {
        background-color: #1f77b4;
        color: white;  /* Keep button text white */
        font-size: 20px;
        border-radius: 12px;
        padding: 12px 30px;
        margin-top: 20px;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #155d8b;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset and model (same as before)
sonar_data = pd.read_csv('sonar_data.csv')
X = sonar_data.drop('Label', axis=1)
Y = sonar_data['Label']
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.4, stratify=None, random_state=1)
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit UI
st.title("Sonar Object Classification")

input_features = []
for feature in X.columns:
    val = st.number_input(f"Input {feature}", value=float(X[feature].mean()))
    input_features.append(val)

input_array = np.array(input_features).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(input_array)
    label = le.inverse_transform(prediction)[0]
    st.write(f"The object is a **{label}**")

# Show training accuracy
train_pred = model.predict(X_train)
acc = accuracy_score(train_pred, Y_train)
st.write(f"Training accuracy: {acc:.2f}")
