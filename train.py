import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("house_data.csv")

# Input and Output
X = data[["District", "Area", "Type", "Floors"]]
y = data["Cost"]

# Categorical and Numeric columns
categorical_features = ["District", "Type"]
numeric_features = ["Area", "Floors"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# AI Model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X, y)

# Save Model
joblib.dump(pipeline, "model.pkl")

print("✅ AI Model Trained Successfully!")
print("Model saved as model.pkl")