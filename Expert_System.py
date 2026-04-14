import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from experta import *
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# FACT
class Patient(Fact):
    """Patient medical data"""
    pass

# EXPERT SYSTEM

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
    def rule_chest_pain(self):
        self.risk += 2

    @Rule(Patient(age=P(lambda x: x > 0.45),
                  chol=P(lambda x: x > 0.30)))
    def rule_age_chol(self):
        self.risk += 2

    @Rule(Patient(exang=1, thalach=P(lambda x: x < 0.45)))
    def rule_angina_hr(self):
        self.risk += 2

    @Rule(Patient(thalach=P(lambda x: x < 0.40)))
    def rule_low_hr(self):
        self.risk += 1

    @Rule(Patient(trestbps=P(lambda x: x > 0.45)))
    def rule_high_bp(self):
        self.risk += 1

    @Rule(Patient(fbs=1))
    def rule_sugar(self):
        self.risk += 1

    @Rule(Patient(sex=1, age=P(lambda x: x > 0.40)))
    def rule_male_age(self):
        self.risk += 1

    @Rule(Patient(cp_1=1, oldpeak=P(lambda x: x < 0.2)))
    def rule_low_risk(self):
        self.risk -= 2

    def get_result(self):
        return 1 if self.risk >= 3 else 0

# MODEL EVALUATION

def evaluate_expert_system(df: pd.DataFrame):

    # 🔥 FIX ADDED HERE
    df = df.replace({True: 1, False: 0})

    X = df.drop("target", axis=1)
    y = df["target"]

    predictions = []

    for _, row in X.iterrows():
        engine = HeartExpert()
        engine.reset()
        engine.declare(Patient(**row.to_dict()))
        engine.run()

        predictions.append(engine.get_result())

    return {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
    }


# USER INPUT SYSTEM

def get_user_input():

    print(" HEART DISEASE PREDICTION SYSTEM")

    data = {
        "age": float(input("Age (0-1 normalized): ")),
        "sex": int(input("Sex (1=male, 0=female): ")),
        "cp_1": int(input("Chest Pain type 1 (0/1): ")),
        "cp_2": int(input("Chest Pain type 2 (0/1): ")),
        "cp_3": int(input("Chest Pain type 3 (0/1): ")),
        "trestbps": float(input("Blood Pressure (0-1): ")),
        "chol": float(input("Cholesterol (0-1): ")),
        "fbs": int(input("Fasting Blood Sugar (0/1): ")),
        "exang": int(input("Exercise Angina (0/1): ")),
        "thalach": float(input("Max Heart Rate (0-1): ")),
        "oldpeak": float(input("Oldpeak (0-1): ")),
        "ca": int(input("Number of major vessels (0-3): "))
    }

    engine = HeartExpert()
    engine.reset()
    engine.declare(Patient(**data))
    engine.run()

    result = engine.get_result()

    print(" RESULT")

    if result == 1:
        print(" HIGH RISK of Heart Disease")
    else:
        print(" LOW RISK of Heart Disease")


# MAIN

if __name__ == "__main__":

    df = pd.read_csv(r"C:/Users/dell/Desktop/university/semester_6/intilligint programming/project/Heart_Disease_Detection/cleaned_data.csv")

    df = df.replace({True: 1, False: 0})

    results = evaluate_expert_system(df)

    print(" MODEL PERFORMANCE")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")

    get_user_input()