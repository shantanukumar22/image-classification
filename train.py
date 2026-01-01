import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# ================================
# 1. Load dataset
# ================================
DATASET_PATH = "dataset.csv"  # <-- change if file name is different

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"{DATASET_PATH} not found! Make sure dataset is in the project folder.")

df = pd.read_csv(DATASET_PATH)

# ================================
# 2. Preprocess
# ================================

# Assume target column is "PriceCategory"
# Example categories: low/medium/high
TARGET_COL = "PriceCategory"

if TARGET_COL not in df.columns:
    raise Exception(f"Target column '{TARGET_COL}' not found in dataset.")

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Identify feature types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Define preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# ================================
# 3. Model pipeline
# ================================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# ================================
# 4. Split data
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 5. Train
# ================================
model.fit(X_train, y_train)

# ================================
# 6. Evaluate
# ================================
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("Accuracy:", accuracy)

# ================================
# 7. Save model.pkl
# ================================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# ================================
# 8. Save metrics.json
# ================================
metrics = {
    "accuracy": float(accuracy)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# ================================
# 9. Save predictions.csv (optional)
# ================================
pred_df = pd.DataFrame({
    "actual": y_test,
    "predicted": preds
})
pred_df.to_csv("predictions.csv", index=False)

print("Training complete. Files created:")
print("- model.pkl")
print("- metrics.json")
print("- predictions.cs")