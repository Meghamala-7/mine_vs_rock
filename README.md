# Mine vs Rock Sonar Classification ⚓🪨

This application predicts whether an underwater object is a Mine or a Rock using sonar signal data and machine learning.

# overview

This web app allows users to input sonar signal features and instantly classify the object as a Mine or Rock using a trained Logistic Regression model.

# Technologies

Frontend:

• Streamlit (interactive UI)

• HTML & CSS (custom styling)

Backend:

• Python

• Pandas, NumPy (data handling)

• Scikit-learn (model training & prediction)

# How to Use
Input the 5 sonar feature values in the provided fields

Click the Predict button

See the result displayed as Mine (💣) or Rock (🪨) immediately

# Requirements

Python 3.x installed

Required Python libraries: pandas, numpy, scikit-learn, streamlit

# COMMANDS

python --version

pip --version

mkdir mine_vs_rock

cd mine_vs_rock

pip install numpy pandas scikit-learn

# Run the Streamlit app

streamlit run app.py

Open your browser and go to:

http://localhost:8501
