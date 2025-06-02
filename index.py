import pandas as pd

# Step 1: Create the dataset
data = pr.read_csv('student_performance.csv')
# Step 2: Load into a DataFrame
df = pd.DataFrame(data)

# Step 3: View the data
print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Define features and target
X = df[['StudyHours', 'Attendance', 'SleepHours', 'PastScore', 'Participation']]
y = df['FinalScore']

# Step 2: Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Actual:", y_test.values)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

import matplotlib.pyplot as plt
import numpy as np

# Plot predicted vs actual scores
x_axis = np.arange(len(y_test))  # [0, 1] for 2 test samples

plt.figure(figsize=(8, 5))
bar_width = 0.35

# Bar for actual scores
plt.bar(x_axis, y_test.values, width=bar_width, label='Actual', color='skyblue')

# Bar for predicted scores
plt.bar(x_axis + bar_width, y_pred, width=bar_width, label='Predicted', color='orange')

# Labels and titles
plt.xlabel('Student Index in Test Set')
plt.ylabel('Final Exam Score')
plt.title('Predicted vs Actual Final Scores')
plt.xticks(x_axis + bar_width / 2, [f'Student {i+1}' for i in x_axis])
plt.legend()
plt.tight_layout()
plt.show()
