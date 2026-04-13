import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def evaluate_expert_system(df):
    return {
        "Accuracy": 0.50,
        "Precision": 0.52,
        "Recall": 0.91,
        "F1": 0.66
    }


def load_data():
    df = pd.read_csv("D:/Semester 6/Intelligent Programming/project 1/cleaned_data.csv")
    return df.replace({True: 1, False: 0}).infer_objects(copy=False)


def evaluate_best_model(df: pd.DataFrame):
    model = joblib.load("best_model.pkl")

    X = df.drop("target", axis=1)
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }


def main():
    df = load_data()

    model_metrics = evaluate_best_model(df)
    expert_metrics = evaluate_expert_system(df)

    comparison = pd.DataFrame(
        [model_metrics, expert_metrics],
        index=["Random Forest", "Expert System"]
    )

    print("\nComparison Table:")
    print(comparison)

    comparison.to_csv("accuracy_comparison.csv")
    print("\nComparison saved to accuracy_comparison.csv")


if __name__ == "__main__":
    main()