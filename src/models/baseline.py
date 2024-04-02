from sklearn.dummy import DummyClassifier
from src.utils import compute_metrics

def flatten(arr):
    return [el for lst_of_el in arr for el in lst_of_el ]

class Baseline:
    def __init__(self, strategy, data, emotions_and_triggers, id2emotions):
        self.strategy = strategy
        self.classifier = DummyClassifier(strategy=strategy)
        self.X = data
        self.Y = emotions_and_triggers
        self.id2emotions = id2emotions


    def score(self):
        out_dict = {'accuracy': None, 'f1-score': {}}
        avg_accuracy = []
        predictions_emotions = []
        predictions_triggers = []

        for column in self.Y.columns:
            y = flatten(self.Y[column])
            x = flatten(self.X)
            self.classifier.fit(x, y)
            predictions = self.classifier.predict(x)
            k = 0
            for text in self.X:
                sentance_list = []
                for sentence in text:
                    sentance_list.append(predictions[k])
                    k += 1
                if column == 'emotions_id':
                    predictions_emotions.append(sentance_list)
                else:
                    predictions_triggers.append(sentance_list)
            avg_accuracy.append(self.classifier.score(x, y))
        out_dict['accuracy'] = round(sum(avg_accuracy)/len(avg_accuracy), 4)
        out_dict['f1-score'] = compute_metrics((predictions_emotions,
                                                self.Y["emotions_id"].values,
                                                predictions_triggers,
                                                self.Y["triggers"].values), self.id2emotions)
        return out_dict