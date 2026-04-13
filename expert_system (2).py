import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from experta import *
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class Patient(Fact):
    pass


class HeartExpert(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.risk = 0

    @Rule(Patient(ca=P(lambda x: x >= 1)))
    def rule_ca(self):
        self.risk += 3

    @Rule(Patient(oldpeak=P(lambda x: x > 0.25)))
    def rule_oldpeak(self):
        self.risk += 3

    @Rule(Patient(cp_2=1) | Patient(cp_3=1))
    def rule_cp(self):
        self.risk += 2

    @Rule(Patient(age=P(lambda x: x > 0.45), chol=P(lambda x: x > 0.30)))
    def rule_age_chol(self):
        self.risk += 2

    @Rule(Patient(exang=1, thalach=P(lambda x: x < 0.45)))
    def rule_angina_hr(self):
        self.risk += 2

    @Rule(Patient(thalach=P(lambda x: x < 0.40)))
    def rule_hr(self):
        self.risk += 1

    @Rule(Patient(trestbps=P(lambda x: x > 0.45)))
    def rule_bp(self):
        self.risk += 1

    @Rule(Patient(fbs=1))
    def rule_sugar(self):
        self.risk += 1

    @Rule(Patient(sex=1, age=P(lambda x: x > 0.40)))
    def rule_male(self):
        self.risk += 1

    @Rule(Patient(cp_1=1, oldpeak=P(lambda x: x < 0.2)))
    def rule_low(self):
        self.risk -= 2

    def get_result(self):
        if self.risk >= 6:
            return 1, "High Risk", self.risk
        elif self.risk >= 3:
            return 1, "Medium Risk", self.risk
        else:
            return 0, "Low Risk", self.risk


def load_data():
    df = pd.read_csv("D:/Semester 6/Intelligent Programming/project 1/cleaned_data.csv")
    df = df.replace({True: 1, False: 0}).infer_objects(copy=False)
    return df


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


def evaluate_expert_system(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    predictions = []

    for _, row in X.iterrows():
        engine = HeartExpert()
        engine.reset()
        engine.declare(Patient(**row.to_dict()))
        engine.run()
        pred, level, score = engine.get_result()
        predictions.append(pred)

    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
    }


def user_input_prediction():
    model = joblib.load("decision_tree_model.pkl")

    data = {}

    data["age"] = float(input("age: "))
    data["sex"] = int(input("sex (0/1): "))
    data["cp"] = int(input("cp (0-3): "))
    data["trestbps"] = float(input("trestbps: "))
    data["chol"] = float(input("chol: "))
    data["fbs"] = int(input("fbs (0/1): "))
    data["restecg"] = int(input("restecg (0-2): "))
    data["thalach"] = float(input("thalach: "))
    data["exang"] = int(input("exang (0/1): "))
    data["oldpeak"] = float(input("oldpeak: "))
    data["slope"] = int(input("slope (0-2): "))
    data["ca"] = int(input("ca (0-4): "))

    model_input = pd.DataFrame([data])
    model_input = pd.get_dummies(model_input)

    if hasattr(model, "feature_names_in_"):
        model_input = model_input.reindex(columns=model.feature_names_in_, fill_value=0)

    model_pred = model.predict(model_input)[0]

    expert_input = pd.get_dummies(pd.DataFrame([data]))
    expert_engine = HeartExpert()
    expert_engine.reset()
    expert_engine.declare(Patient(**expert_input.iloc[0].to_dict()))
    expert_engine.run()
    expert_pred, risk_level, risk_score = expert_engine.get_result()

    print("\nPrediction Result")
    print("Decision Tree Prediction:", model_pred)

    if model_pred == 1:
        print("Decision Tree Result: Patient may have heart disease")
    else:
        print("Decision Tree Result: Patient may not have heart disease")

    print("Expert System Prediction:", expert_pred)
    print("Expert System Risk Level:", risk_level)
    print("Expert System Risk Score:", risk_score)

    if risk_score >= 6:
        print("Estimated condition: about 80% - 90% risk")
    elif risk_score >= 3:
        print("Estimated condition: about 50% - 70% risk")
    else:
        print("Estimated condition: about 10% - 30% risk")


def main():
    df = load_data()

    model_metrics = evaluate_model(df)
    expert_metrics = evaluate_expert_system(df)

    comparison = pd.DataFrame(
        [model_metrics, expert_metrics],
        index=["Decision Tree", "Expert System"]
    )

    print(comparison)

    user_input_prediction()

    comparison.to_csv("accuracy_comparison.csv")


if __name__ == "__main__":
    main()