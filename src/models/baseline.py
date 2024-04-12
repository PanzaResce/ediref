from sklearn.dummy import DummyClassifier
from src.utils import compute_metrics
from transformers import EvalPrediction
def flatten(arr):
    return [el for lst_of_el in arr for el in lst_of_el ]

class Baseline:
    def __init__(self, strategy, data, emotions_and_triggers):
        self.strategy = strategy
        self.classifier = DummyClassifier(strategy=strategy)
        self.X = data
        self.Y = emotions_and_triggers


    def score(self):
        out_dict = {'accuracy': None, 'f1-score': {}}
        predictions_emotions = []
        predictions_triggers = []
        predictions = [predictions_emotions,
                       predictions_triggers]

        for i, column in enumerate(self.Y.column_names):
            self.classifier.fit(self.X, self.Y[column])
            predictions[i] = self.classifier.predict(self.X )
        
        label_ids = [self.Y["emotions_id"], self.Y["triggers"], self.X]
        eval_pred =EvalPrediction(predictions=predictions, label_ids=label_ids) 
            
        out_dict = compute_metrics(eval_pred)
        return out_dict