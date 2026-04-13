import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv("D:/Semester 6/Intelligent Programming/project 1/cleaned_data.csv")
    df = df.replace({True: 1, False: 0}).infer_objects(copy=False)
    return df


def expert_predict(row):
    high = 0
    medium = 0
    low = 0

    if row.get("chol", 0) > 0.26:
        high += 1
    if row.get("age", 0) > 0.43 and row.get("trestbps", 0) > 0.43:
        high += 1
    if row.get("thalach", 0) < 0.37:
        medium += 1
    if row.get("oldpeak", 0) > 0.32:
        high += 1
    if row.get("exang", 0) == 1:
        medium += 1
    if row.get("fbs", 0) == 1:
        medium += 1
    if row.get("sex", 0) == 1 and row.get("age", 0) > 0.33:
        medium += 1
    if row.get("ca", 0) >= 2:
        high += 1
    if row.get("cp_1", 0) == 1:
        low += 1
    if row.get("slope_2", 0) == 1:
        medium += 1

    score = high * 3 + medium * 2 - low
    if score >= 4:
        return 1
    else:
        return 0


def evaluate_model(df):
    model = joblib.load("decision_tree_model.pkl")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if hasattr(model, "feature_names_in_"):
        X_test = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

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
        index=["Decision Tree", "Expert System"]
    )

    print(comparison)

    comparison.to_csv("accuracy_comparison.csv")


if __name__ == "__main__":
    main()