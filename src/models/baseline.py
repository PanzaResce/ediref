from sklearn.dummy import DummyClassifier
from src.utils import f1_score_on_flat
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
            'f1scores_emotions_flatten': None,
            'f1scores_triggers_flatten': None
        }

        self.classifier.fit(self.X, self.Y["emotions_id"])
        out_dict["accuracy_emotions"] = self.classifier.score(self.X, self.Y["emotions_id"])      
        out_dict["f1scores_emotions_flatten"] = f1_score_on_flat(self.X, self.Y["emotions_id"], list(range(7)))
        
        self.classifier.fit(self.X, self.Y["triggers"])
        out_dict["accuracy_triggers"] = self.classifier.score(self.X, self.Y["triggers"])      
        out_dict["f1scores_triggers_flatten"] = f1_score_on_flat(self.X, self.Y["triggers"], list(range(7)))

        return out_dict