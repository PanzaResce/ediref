import pandas as pd
from datasets import Dataset
import torch
from torch import Tensor
from torch import randint
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix, BinaryConfusionMatrix, MulticlassPrecisionRecallCurve, BinaryPrecisionRecallCurve, MulticlassF1Score, BinaryF1Score, MulticlassAccuracy, BinaryAccuracy
import re
import numpy as np
from enum import auto
import matplotlib.pyplot as plt
class ErrorAnalysis:
    
    

    def __init__(self, df, emotion_labels=['0','1','2','3','4','5','6'], trigger_labels=['0','1']):
        self.__counter = 0
        self.separator = '[SEP]'
        self.METRICS = [ "accuracy", "accuracy_none", "f1_macro", "f1_micro", "f1_weighted", "f1_none"]
        if  isinstance(df, Dataset):
            self.df = self.dataset2dataframe(df)
        else: self.df = df
        if type(self.df['emotions_id'].iloc[0]) == list:
            self.df['emotions_id'] = self.df['emotions_id'].map(lambda x: x.index(max(x))) #change one-hot encoding to normal
        self.df['utterance_text'] = self.get_utterance_column(self.df)
        self.features=self.df.columns.values.tolist()
        self.emotion_labels = emotion_labels
        self.trigger_labels = trigger_labels


    def dataset2dataframe(self, ds):
        ds.set_format()
        return pd.DataFrame.from_records([r for r in ds])
    
    def add_model(self, trainer, ds, model_name='', seed = 42, pred_type=None):
        assert ds.shape[0] == self.df.shape[0], f"Wrong Dataset passed! it has {ds.shape[0]} rows, instead of {self.df.shape[0]}" 
        #get model predictions
        try:
            outputs = trainer.predict(ds).predictions
            logits = torch.tensor(outputs[0])
            triggers = torch.tensor(outputs[1])
            #print(logits.shape, triggers.shape)
        except:
            print("cannot run model(ds)")
        s1 = self.add_predictions(logits, model_name, seed, pred_type= "emotions")
        s2 = self.add_predictions(triggers, model_name, seed, pred_type="trigger")
        return(f"Successfullty added model!({s1}, {s2})")
    
    def add_predictions(self, logits, model_name='', seed = 42, pred_type=None):
        if not pred_type and model_name[-2:]=='_t': pred_type = 'trigger'
        elif not pred_type: pred_type = 'emotions'
        assert pred_type == 'emotions' or pred_type == 'trigger', "Wrong pred_type string passed. Pass 'trigger' or 'emotions'"
        if isinstance(logits,Tensor):
            if pred_type == 'emotions': logits = logits.softmax(dim=-1)
            logits = logits.unbind(dim=0)
        assert len(logits) == self.df.shape[0], f"Wrong predictions passed! They are {len(logits)} rows, instead of {self.df.shape[0]}" 
        if model_name == '': 
            model_name = f"model{self.__counter}"
            self.__counter = self.__counter + 1
        match = re.search("_\d+", model_name) != None
        if not match: model_name = model_name.replace("_t",'') + f"_{seed}"
        if pred_type=='trigger' and model_name[-2:]!='_t': model_name = model_name + "_t"
        #add logits to the dataframe !WE WANT A LIST OF len C tensors! (Batch of Utterances x classes)
        self.df[model_name] = logits
        return (f"Successfullty added predictions for model {model_name}!")
    

    def get_confusion_matrix(self, model_name, seed = 42, pred_type=None, plot = True):
        if not pred_type and model_name[-2:]=='_t': pred_type = 'trigger'
        elif not pred_type: pred_type = 'emotions'
        assert pred_type == 'emotions' or pred_type == 'trigger', "Wrong pred_type string passed. Pass 'trigger' or 'emotions'"
        #check if it exits model
        extra = "_t" if pred_type == "trigger" else ""
        if f"{model_name}_{seed}{extra}" in self.df.columns.values:
            model_name = f"{model_name}_{seed}{extra}"
        if model_name not in self.df.columns.values:
            raise NameError(f"Not found model {model_name}")
        
        #already exploded df

        #confusion_matrix want preds (N,...) and target(N, ...), same dimensions
        preds = torch.stack(self.df[model_name].values.tolist())
        if pred_type == "emotions":
            preds = preds.argmax(dim = -1)
            target = torch.tensor(self.df['emotions_id'].values)
            metric = MulticlassConfusionMatrix(num_classes=7, normalize='true')
            labels = self.emotion_labels
        else :
            target = torch.tensor(self.df['triggers'].values)[:,None]
            metric = BinaryConfusionMatrix(normalize='all')
            labels = self.trigger_labels
        metric.update(preds, target)
        if plot:
            fig_, ax_ = metric.plot()
            ax_.set_xticklabels(labels)
            ax_.set_yticklabels(labels)
            plt.show()
        return metric.compute()
    
    def get_precision_recall(self, model_name, seed = 42, pred_type=None, plot = True):
        if not pred_type and model_name[-2:]=='_t': pred_type = 'trigger'
        elif not pred_type: pred_type = 'emotions'
        assert pred_type == 'emotions' or pred_type == 'trigger', "Wrong pred_type string passed. Pass 'trigger' or 'emotions'"
        #check if it exits model
        extra = "_t" if pred_type == "trigger" else ""
        if f"{model_name}_{seed}{extra}" in self.df.columns.values:
            model_name = f"{model_name}_{seed}{extra}"
        if model_name not in self.df.columns.values:
            raise NameError(f"Not found model {model_name}")
        #aldready exploded df

        #precision recal want preds (N,C,...) and target(N, ...) with target has one less dimension
        preds = torch.stack(self.df[model_name].values.tolist())
        if pred_type == "emotions":
            target = torch.tensor(self.df['emotions_id'].values)
            metric = MulticlassPrecisionRecallCurve(num_classes=7, thresholds = list(np.linspace(0,1,20)))
            labels = self.emotion_labels
        else :
            target = torch.tensor(self.df['triggers'].values)[:,None]
            metric = BinaryPrecisionRecallCurve(thresholds = list(np.linspace(0,1,20)))
            labels = self.trigger_labels
        metric.update(preds, target)
        if plot:
            fig_, ax_ = metric.plot()
            leg = plt.legend(labels)
            plt.show()
        return metric.compute()
    


    def get_performance_table(self, metrics = ["accuracy", "f1_macro"]):
        emotion_target = torch.tensor(self.df['emotions_id'].values)
        trigger_target = torch.tensor(self.df['triggers'].values)[:, None]
        assert all([(metric in self.METRICS) for metric in metrics]), "Not all metrics passed are valid. Pass only metrics present in {a}".format(a=[ "accuracy", "f1_macro", "f1_micro", "f1_weighted"])

        outputs = self.df.drop(columns= self.features)
        emotion_collection = MetricCollection([])
        if "accuracy" in metrics: emotion_collection.add_metrics({"accuracy": MulticlassAccuracy(num_classes=7, average='micro')})
        if "f1_micro" in metrics: emotion_collection.add_metrics({"f1_micro": MulticlassF1Score(num_classes=7, average= 'micro')})
        if "f1_macro" in metrics: emotion_collection.add_metrics({"f1_macro": MulticlassF1Score(num_classes=7, average= 'macro')})
        if "f1_weighted" in metrics: emotion_collection.add_metrics({"f1_weighted": MulticlassF1Score(num_classes=7, average= 'weighted')})
        if "f1_none" in metrics: emotion_collection.add_metrics({"f1": MulticlassF1Score(num_classes=7, average= 'none')})
        if "accuracy_none" in metrics: emotion_collection.add_metrics({"accuracy": MulticlassAccuracy(num_classes=7, average='none')})
        trigger_collection = MetricCollection([])
        if "accuracy" in metrics: trigger_collection.add_metrics({"accuracy_t": BinaryAccuracy()})
        if any(f1 in metrics for f1 in ["f1_micro","f1_macro","f1_weighted"]): trigger_collection.add_metrics({"f1_t": BinaryF1Score()})

        res = {}
        for model_name in outputs.columns.values:
            preds = torch.stack(outputs[model_name].values.tolist())
            pred_type = "emotions"
            if model_name[-2:] == '_t':
                model_name = model_name[:-2]
                pred_type = "trigger"
            if pred_type == "trigger":
                metrics = trigger_collection(preds, trigger_target)
            else:
                preds = preds.argmax(dim = -1)
                metrics = emotion_collection(preds, emotion_target)
            res[model_name] = res.get(model_name, {'name': model_name})
            res[model_name].update(metrics)
        return res
    
    def get_misclassified(self, model_name, seed = 42, pred_type=None, t_threshold = 0.5):
        if not pred_type and model_name[-2:]=='_t': pred_type = 'trigger'
        elif not pred_type: pred_type = 'emotions'
        assert pred_type == 'emotions' or pred_type == 'trigger', "Wrong pred_type string passed. Pass 'trigger' or 'emotions'"
        #check if it exits model
        extra = "_t" if pred_type == "trigger" else ""
        if f"{model_name}_{seed}{extra}" in self.df.columns.values:
            model_name = f"{model_name}_{seed}{extra}"
        if model_name not in self.df.columns.values:
            raise NameError(f"Not found model {model_name}")
        selector = "emotions_id" if pred_type == "emotions" else "triggers"
        preds = self.df[model_name].map(lambda x: x.argmax()) if pred_type == "emotions" else self.df[model_name].map(lambda x: x>t_threshold)
        frame = self.df[self.df[selector] != preds]
        return frame[self.features + [model_name]]
    
    def get_emotions_table(self):
        #get accuracy or F1-score of each emotions on each model.
        support = self.df['emotions_id'].value_counts().to_dict()
        support = {(self.emotion_labels[k]+metr): support[k] for k in support.keys() for metr in ["_acc","_f1"] }
        support['name'] = 'support'
        table = self.get_performance_table(metrics = ["accuracy_none", "f1_none"])
        for model_name, metrics in table.items():
            if 'accuracy' in table[model_name].keys() :
                table[model_name].update({self.emotion_labels[i]+'_acc': v for i,v in enumerate(metrics['accuracy'])})
                del table[model_name]['accuracy']
            if 'f1' in table[model_name].keys() :
                table[model_name].update({self.emotion_labels[i]+'_f1': v for i,v in enumerate(metrics['f1'])})
                del table[model_name]['f1']
        table['support'] = support
        return table
    
    def get_utterance_column(self, x):
        """
        x: row of a pandas dataframe [Series] or Dataframe itself [Dataframe]
        """
        if type(x) == pd.Series:
            assert 'dialogue_text' in x.index.values, "Not found property 'dialogue_text' for row series!" 
            assert 'utterance_index' in x.index.values, "Not found property 'utterance_index' for row series!" 
            splits = x['dialogue_text'].split(self.separator)
            return splits[x['utterance_index']]
        elif type(x) == pd.DataFrame:
            assert 'dialogue_text' in x.columns.values, "Not found column 'dialogue_text'!" 
            assert 'utterance_index' in x.columns.values, "Not found column 'utterance_index'!" 
            splits = x.apply(lambda row: row['dialogue_text'].split(self.separator)[row['utterance_index']], axis = 1)
        return splits
    

    def get_utterance_ranking(self,  model_name, seed = 42, pred_type=None, t_threshold = 0.5):
        if not pred_type and model_name[-2:]=='_t': pred_type = 'trigger'
        elif not pred_type: pred_type = 'emotions'
        assert pred_type == 'emotions' or pred_type == 'trigger', "Wrong pred_type string passed. Pass 'trigger' or 'emotions'"
        #check if it exits model
        extra = "_t" if pred_type == "trigger" else ""
        if f"{model_name}_{seed}{extra}" in self.df.columns.values:
            model_name = f"{model_name}_{seed}{extra}"
        if model_name not in self.df.columns.values:
            raise NameError(f"Not found model {model_name}")
        
        def accuracy_by_group(model_name, t_threshold= 0.5): #function that generated the aggreagator
            if model_name[-2:] == '_t':
                target_label = 'triggers'
                def pred(x): 
                    return x>t_threshold
            else:
                target_label = 'emotions_id'
                def pred(x): 
                    return x.argmax()
                
            def func(df): #aggreagator definition
                corrects = df.apply(lambda x: int(x[target_label] == pred(x[model_name])), axis = 1)
                lengths = df.apply(lambda x: int(x[target_label] == pred(x[model_name])), axis = 1)
                return pd.Series({'accuracy': corrects.agg('mean'), 'dialogue_len': df.shape[0]})
            
            return func
        
        return self.df.groupby(['episode']).apply(
            accuracy_by_group(model_name, t_threshold)
        ).sort_values(['accuracy','dialogue_len'], ascending=[True, False])
    
    def get_trigger_effect(self, model_name, seed = 42, t_threshold = 0.5):
        #check if it exits model
        extra = "_t"
        if model_name[-2:] == '_t': model_name=model_name[:-2]
        if f"{model_name}_{seed}" in self.df.columns.values or f"{model_name}_{seed}{extra}" in self.df.columns.values:
            model_name = f"{model_name}_{seed}"
        if model_name not in self.df.columns.values:
            raise NameError(f"Not found model {model_name}")
        if (model_name + extra) not in self.df.columns.values:
            raise NameError(f"Not found trigger results for model {model_name}")
        
        correct_triggers = self.df[(self.df['triggers']==1) & (self.df[model_name+extra].map(lambda x: x>t_threshold))]
        correct_triggers = (correct_triggers['utterance_index']+1).astype(str)+correct_triggers['episode']
        bad_triggers = self.df[(self.df['triggers']==1) & (self.df[model_name+extra].map(lambda x: x<=t_threshold))]
        bad_triggers = (bad_triggers['utterance_index']+1).astype(str)+bad_triggers['episode']
        support = [correct_triggers.shape[0], bad_triggers.shape[0]]
        def score(x):
            return x['emotions_id']==x[model_name].argmax(dim = -1).item()
        accuracy = [
            self.df[(self.df['utterance_index'].astype(str)+self.df['episode']).isin(correct_triggers)].apply(score, axis = 1).mean(),
            self.df[(self.df['utterance_index'].astype(str)+self.df['episode']).isin(bad_triggers)].apply(score, axis = 1).mean()
        ]
        if support[0] == 0.: accuracy[0] = 0.
        if support[1] == 0.: accuracy[1] = 0.

        return {model_name:{ 'emotion_accuracy':{'correct_t': accuracy[0], 'bad_t': accuracy[1]},
                            'support':{'correct_t': support[0], 'bad_t': support[1]}}}
    
    def get_trigger_table(self, t_threshold = 0.5):
        table = {}
        for col_name in self.df.columns.values:
            if col_name not in self.features and col_name[-2:]!='_t' and col_name+'_t' in self.df.columns.values:
                table.update(self.get_trigger_effect(col_name, t_threshold=t_threshold))

        return table


    


