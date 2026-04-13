import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_data():
    df = pd.read_csv("D:/Semester 6/Intelligent Programming/project 1/cleaned_data.csv")
    return df.replace({True: 1, False: 0}).infer_objects(copy=False)


def select_features(df):
    selected = [
        "cp_2", "cp_3",
        "ca",
        "exang",
        "oldpeak",
        "thalach",
        "chol",
        "trestbps",
        "age",
        "sex",
        "target"
    ]

    return df[[col for col in selected if col in df.columns]]


def train_decision_tree(df):
    df = select_features(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best", "random"],
        "max_depth": [3, 4, 5, 6, 8, 10, None],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": [None, "sqrt", "log2"],
        "class_weight": [None, "balanced"],
        "ccp_alpha": [0.0, 0.001, 0.005, 0.01]  
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    y_pred = model.predict(X_test)

    print("\n BEST PARAMETERS:")
    print(grid.best_params_)

    print("\n DECISION TREE RESULTS:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    joblib.dump(model, "decision_tree_model.pkl")

    return model


if __name__ == "__main__":
    df = load_data()
    train_decision_tree(df)