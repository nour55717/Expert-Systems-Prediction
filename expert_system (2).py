import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from experta import *
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


    @Rule(Patient(age=P(lambda x: x > 0.45),
                  chol=P(lambda x: x > 0.30)))
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

    # ------------------
    def get_result(self):
        return 1 if self.risk >= 3 else 0


def evaluate_expert_system(df: pd.DataFrame):

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

if __name__ == "__main__":
    df = pd.read_csv("D:/Semester 6/Intelligent Programming/project 1/cleaned_data.csv")
    df = df.replace({True: 1, False: 0}).infer_objects(copy=False)

    results = evaluate_expert_system(df)

    print("\n Expert System Results:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")