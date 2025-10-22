import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# 1. Muat data
script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(script_dir, "namadataset_preprocessing", "pima_diabetes_preprocessing.csv")

# Muat data
df = pd.read_csv(data_path)

# 2. Pisahkan fitur (X) dan target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Aktifkan MLflow Autolog
mlflow.sklearn.autolog()

# 5. Latih model di dalam 'run'
with mlflow.start_run():
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_preds = model.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    
    print(f"Akurasi Model (CI Run): {acc}")

print("MLflow run finished.")