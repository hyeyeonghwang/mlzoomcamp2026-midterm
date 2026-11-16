
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1. Load data
df = pd.read_csv("caffeine_crash_and_anxiety_telemetry.csv")

numeric_features = [
    "age", "baseline_anxiety_score", "prior_sleep_hours",
    "daily_hydration_liters", "caffeine_mg",
    "sugar_grams", "consumption_duration_min"
]

categorical_features = [
    "gender", "cyp1a2_metabolizer_type",
    "caffeine_tolerance", "primary_caffeine_source"
]

features = numeric_features + categorical_features

X = df[features]
y = df["crash_risk_level"]

# 2. Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# 5. Final model
# Replace with the selected model and parameters from notebook
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])

# 6. Train
model.fit(X_train, y_train)

print("Test accuracy:", round(model.score(X_test, y_test), 4))

# 7. Save
Path("model").mkdir(exist_ok=True)

joblib.dump({
    "model": model,
    "classes": encoder.classes_.tolist(),
    "features": features
}, "model/model.pkl")

print("Model saved: model/model.pkl")
