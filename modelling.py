import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    mlflow.set_tracking_uri("file:./mlruns")

    DATA_PATH = "heart_disease_preprocessing.csv"

    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run(run_name="rf_autolog_basic"):
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy_manual", acc)

        mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training selesai: modelling.py")

if __name__ == "__main__":
    main()
