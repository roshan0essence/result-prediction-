import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# --- Model training (use your same dataset)
import pandas as pd

data = pd.read_csv('student_performance.csv')

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
