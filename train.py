# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Simulate data loading
print("Starting data preparation...")
data = {'feature1': [1, 2, 3, 4, 10],
        'feature2': [6, 7, 8, 9, 10],
        'target': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

X = df[['feature1', 'feature2']]
y = df['target']

# 2. Simulate model training
print("Training model...")
model = LogisticRegression()
model.fit(X, y)

# 3. Simulate model evaluation/success
score = model.score(X, y)
print(f"Model trained successfully! Accuracy: {score:.2f}")

# The pipeline will look for this output to confirm success
print("PIPELINE_SUCCESS")
