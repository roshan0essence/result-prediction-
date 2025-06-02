import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)

# Generate 100 random entries
data = {
    'StudyHours': np.random.normal(loc=3.5, scale=1.5, size=5000).clip(1, 7),  # Hours between 1-7
    'Attendance': np.random.normal(loc=80, scale=15, size=5000).clip(40, 100).astype(int),  # 40-100%
    'SleepHours': np.random.choice([5, 6, 7, 8], size=5000, p=[0.1, 0.3, 0.4, 0.2]),  # Discrete values
    'PastScore': np.random.normal(loc=70, scale=15, size=5000).clip(30, 95).astype(int),  # 30-95
    'Participation': np.random.binomial(n=1, p=0.7, size=5000),  # Binary (0 or 1)
    'FinalScore': None  # Will be calculated
}

# Simulate FinalScore as a weighted combination of other factors
data['FinalScore'] = (
    0.3 * data['PastScore'] + 
    0.25 * (data['StudyHours'] * 10) + 
    0.2 * data['Attendance'] + 
    0.15 * (data['SleepHours'] * 10) + 
    0.1 * (data['Participation'] * 20)
).astype(int).clip(30, 100)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('student_performance.csv', index=False)
print("CSV file 'student_performance.csv' created with 100 entries!")