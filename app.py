import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# --- Model training (use your same dataset)
import pandas as pd

data = {
    'StudyHours': [4, 2, 5, 1, 3, 6, 2.5, 4.5, 1.5, 5.5],
    'Attendance': [90, 60, 95, 50, 75, 98, 70, 85, 55, 96],
    'SleepHours': [7, 6, 8, 5, 6, 7, 6, 7, 5, 8],
    'PastScore': [75, 55, 82, 40, 65, 90, 60, 78, 45, 88],
    'Participation': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    'FinalScore': [80, 58, 85, 45, 70, 92, 63, 82, 50, 90]
}

df = pd.DataFrame(data)
X = df[['StudyHours', 'Attendance', 'SleepHours', 'PastScore', 'Participation']]
y = df['FinalScore']
model = LinearRegression()
model.fit(X, y)

# --- Streamlit App UI
st.title("📘 Student Exam Score Predictor")

st.write("Enter the student's details:")

study_hours = st.slider('Study Hours Per Day', 0.0, 10.0, 3.0)
attendance = st.slider('Attendance (%)', 0, 100, 75)
sleep_hours = st.slider('Sleep Hours Per Night', 0.0, 12.0, 7.0)
past_score = st.slider('Past Exam Score', 0, 100, 60)
participation = st.radio('Class Participation', ['Yes', 'No'])

# Convert participation to binary
participation_value = 1 if participation == 'Yes' else 0

# Predict button
if st.button("Predict Final Score"):
    input_data = np.array([[study_hours, attendance, sleep_hours, past_score, participation_value]])
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Final Score: {prediction[0]:.2f}")
