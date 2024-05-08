from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
def flatten(arr):
    return [el for lst_of_el in arr for el in lst_of_el ]

class Baseline:
    def __init__(self, strategy, data, emotions_and_triggers):
        self.strategy = strategy
        self.classifier = DummyClassifier(strategy=strategy)
        self.X = data
        self.Y = emotions_and_triggers


    def score(self):
        out_dict = {
            'accuracy_emotions': None,
            'accuracy_triggers': None,
            'f1scores_emotions': None,
            'f1scores_triggers': None
        }

        self.classifier.fit(self.X, self.Y["emotions_id"])
        out_dict["accuracy_emotions"] = self.classifier.score(self.X, self.Y["emotions_id"])      
        out_dict["f1scores_emotions"] = round(f1_score(self.Y["emotions_id"], self.classifier.predict(self.X), average='macro', labels=list(range(7)), zero_division=0), 4)
        
        
        self.classifier.fit(self.X, self.Y["triggers"])
        out_dict["accuracy_triggers"] = self.classifier.score(self.X, self.Y["triggers"])      
        out_dict["f1scores_triggers"] = round(f1_score(self.Y["triggers"], self.classifier.predict(self.X), average='macro', labels=list(range(7)), zero_division=0), 4)

        out_dict["u_avg_f1"] = round((out_dict["f1scores_emotions"] + out_dict["f1scores_triggers"]) /2, 4)

        return out_dict