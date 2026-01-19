import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Inputs: [Hours studied, Attendance]
X = np.array([
    [1, 40],
    [2, 50],
    [3, 60],
    [4, 75],
    [5, 80],
    [6, 90]
])

# Output: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])

# Model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict for new student
new_student = np.array([[3, 72]])
prediction = model.predict(new_student)

print("Prediction (0=Fail, 1=Pass):", prediction[0])
