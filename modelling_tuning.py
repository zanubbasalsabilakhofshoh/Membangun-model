
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

# Load dataset hasil preprocessing
df = pd.read_csv("heart_disease_preprocessing.csv")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    scoring='accuracy'
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Manual logging
with mlflow.start_run(run_name="rf_tuning_advanced"):
    mlflow.log_param("n_estimators", best_model.n_estimators)
    mlflow.log_param("max_depth", best_model.max_depth)

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Artefak tambahan
    with open("feature_info.txt", "w") as f:
        f.write(str(X.columns.tolist()))

    mlflow.log_artifact("feature_info.txt")
    mlflow.sklearn.log_model(best_model, "best_model")

print("Training selesai: modelling_tuning.py")
