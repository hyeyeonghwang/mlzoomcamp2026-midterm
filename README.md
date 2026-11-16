# Caffeine Crash Risk Prediction

## 1. Problem Description

Individuals experience different levels of fatigue, anxiety, and other symptoms after consuming caffeine.

This project aims to predict individual caffeine crash risk using demographic characteristics, caffeine consumption patterns, and biological factors.

The task is a **multiclass classification problem** with four risk levels:

- Sustained / Negligible Crash
- Mild Afternoon Fatigue
- Moderate Crash & Lethargy
- Severe Crash & Tachycardia

The project also investigates which features contribute most to the model's predictions.

## 2. Dataset

The dataset contains 8,000 observations and 18 variables. (Kaggle: https://www.kaggle.com/datasets/samartalwar/caffeine-crash-and-jitters-energy-drinks-and-anxiety?resource=download)

Eleven input features are selected, including age, caffeine intake, sleep duration, CYP1A2 metabolizer type, caffeine tolerance, and other individual characteristics.

Post-consumption symptoms and `crash_severity_score` are excluded from the predictors to prevent data leakage.

Dataset file: `caffeine_crash_and_anxiety_telemetry.csv`

## 3. Machine Learning Pipeline

The following models are compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Categorical variables are encoded using OneHotEncoder, and numerical variables are standardized using StandardScaler.

The dataset is divided into training (60%), validation (20%), and test (20%) sets.

Model selection is based on validation Macro F1-score. Hyperparameters are tuned using RandomizedSearchCV.

The selected model is evaluated on the held-out test set.

Permutation feature importance is used to interpret model predictions.

## 4. Project Structure

```text
mlzoomcamp2026-midterm/
├── data/
    └── caffeine_crash_and_anxiety_telemetry.csv
├── notebook.ipynb
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── README.md
└── model/
    └── model.pkl
```

## 5. Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Start the prediction service:

```bash
python predict.py
```

The API is available at `http://localhost:9696/predict`.

## 6. Run with Docker

Build the Docker image:

```bash
docker build -t caffeine-crash .
```

Run the container:

```bash
docker run --rm -p 9696:9696 caffeine-crash
```

## 7. Test the API

Send a POST request:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 29,
    "gender": "Female",
    "cyp1a2_metabolizer_type": "Slow",
    "caffeine_tolerance": "Low",
    "baseline_anxiety_score": 5,
    "prior_sleep_hours": 5.5,
    "daily_hydration_liters": 1.5,
    "primary_caffeine_source": "Brewed Drip Coffee",
    "caffeine_mg": 200,
    "sugar_grams": 10,
    "consumption_duration_min": 20
  }'
```

The API returns the predicted crash risk category and class probabilities.

## 8. Limitations

Feature importance describes predictive associations, not causal relationships.

The dataset's provenance and the method used to generate crash severity labels should be verified before drawing biological conclusions.

## 9. Technologies

Python, pandas, scikit-learn, XGBoost, Flask, Gunicorn, Docker.
