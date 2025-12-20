import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Heart Disease Classification")

    DATA_PATH = "heart_disease_preprocessing.csv"
    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="rf_autolog"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

if __name__ == "__main__":
    main()
