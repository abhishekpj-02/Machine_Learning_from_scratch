import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  
y = np.array([0, 0, 0, 1, 1, 1])                


model = LogisticRegression()
model.fit(X, y)

# Prediction
new_student = np.array([[3.5]])
probability = model.predict_proba(new_student)
prediction = model.predict(new_student)

print("Probability of passing:", probability[0][1])
print("Prediction (0=Fail, 1=Pass):", prediction[0])
