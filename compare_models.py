import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv("D:/Semester 6/Intelligent Programming/project 1/cleaned_data.csv")
    return df.replace({True: 1, False: 0}).infer_objects(copy=False)


def expert_predict(row):
    high = 0
    medium = 0
    low = 0

    # normalized rules
    if row["chol"] > 0.26:
        high += 1
    if row["age"] > 0.43 and row["trestbps"] > 0.43:
        high += 1
    if row["thalach"] < 0.37:
        medium += 1
    if row["oldpeak"] > 0.32:
        high += 1
    if row["exang"] == 1:
        medium += 1
    if row["fbs"] == 1:
        medium += 1
    if row["sex"] == 1 and row["age"] > 0.33:
        medium += 1
    if row["ca"] >= 2:
        high += 1
    if "cp_1" in row and row["cp_1"] == 1:
        low += 1
    if "slope_2" in row and row["slope_2"] == 1:
        medium += 1

    score = high * 3 + medium * 2 - low
    return 1 if score >= 4 else 0


def evaluate_model(df):
    model = joblib.load("best_model.pkl")

    X = df.drop("target", axis=1)
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }


def evaluate_expert(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    preds = X.apply(expert_predict, axis=1)

    return {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds, zero_division=0),
        "Recall": recall_score(y, preds, zero_division=0),
        "F1": f1_score(y, preds, zero_division=0),
    }


def main():
    df = load_data()

    model_metrics = evaluate_model(df)
    expert_metrics = evaluate_expert(df)

    comparison = pd.DataFrame(
        [model_metrics, expert_metrics],
        index=["Random Forest", "Expert System"]
    )

    print("\n MODEL COMPARISON:")
    print(comparison)

    comparison.to_csv("accuracy_comparison.csv")


if __name__ == "__main__":
    main()