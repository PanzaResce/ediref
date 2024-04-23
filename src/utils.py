import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import urllib
import tqdm
from datasets import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sklearn.metrics import f1_score, accuracy_score

def set_seeds(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def download_url(download_path: Path, url: str):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    print(url)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=download_path, reporthook=t.update_to)

def f1_score_per_instance(y_pred, y_true, labels):
    scores = [f1_score(y_true[i], y_pred[i], average='macro', labels=labels, zero_division=0) for i in range(len(y_pred))]
    return round(sum(scores)/len(scores), 4)

def f1_score_on_flat(y_pred, y_true, labels):
    result = round(f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0), 4)
    return result

def classify_emotion_logits(predicted_emotion_logits, thresh=0.5):
    predicted_emotion_logits = (predicted_emotion_logits > thresh)
    index = [i for i,x in enumerate(predicted_emotion_logits) if x]
    return index[0]


# def compute_metrics_for_trainer(eval_pred):
#     [flat_pred_emotions_logits, flat_pred_triggers_logits] = eval_pred.predictions
#     flat_pred_emotions = []
#     flat_pred_triggers = torch.round(F.sigmoid(torch.from_numpy(flat_pred_triggers_logits)))
#     flat_pred_emotions = torch.argmax(F.softmax(torch.from_numpy(flat_pred_emotions_logits), -1), -1)
    # for i, predicted_emotion_logits in enumerate(flat_pred_emotions_logits):
        # flat_pred_emotions.append(classify_emotion_logits(torch.tensor(predicted_emotion_logits)))
        # flat_pred_emotions.append(torch.argmax(F.softmax(torch.from_numpy(predicted_emotion_logits), -1)).item())
        # flat_pred_triggers.append((flat_pred_triggers_logits[i]>0.5)[0])
    # eval_pred.predictions = (flat_pred_emotions, flat_pred_triggers)
    # return compute_metrics(eval_pred)
    

def restore_texts(eval_pred):
    [flat_pred_emotions, flat_pred_triggers] = eval_pred.predictions
    [flat_label_emotions, flat_label_triggers, dialogue_index] = eval_pred.label_ids[0:3]
    index_of_last_text = -1
    k = -1
    predictions_emotions, labels_emotions, predictions_triggers, labels_triggers = [], [], [], []
    for i, index_of_dialog in enumerate(dialogue_index):
        if not index_of_dialog == index_of_last_text:
            predictions_emotions.append([flat_pred_emotions[i]])
            labels_emotions.append([flat_label_emotions[i]])
            predictions_triggers.append([flat_pred_triggers[i]])
            labels_triggers.append([flat_label_triggers[i]])
            index_of_last_text = index_of_dialog
            k+=1
        else:
            predictions_emotions[k].append(flat_pred_emotions[i])
            labels_emotions[k].append(flat_label_emotions[i])
            predictions_triggers[k].append(flat_pred_triggers[i])
            labels_triggers[k].append(flat_label_triggers[i])
    return predictions_emotions, labels_emotions, predictions_triggers, labels_triggers


def compute_metrics(eval_pred, emotion_count=7):
    [pred_emotions_logits, pred_triggers_logits] = eval_pred.predictions
    [flat_label_emotions, flat_label_triggers] = eval_pred.label_ids[0:2]

    flat_pred_triggers = torch.round(F.sigmoid(torch.from_numpy(pred_triggers_logits)))
    flat_pred_emotions = torch.argmax(F.softmax(torch.from_numpy(pred_emotions_logits), -1), -1)
    
    # predictions_emotions, labels_emotions, predictions_triggers, labels_triggers = restore_texts(eval_pred)

    # f1scores_emotions_instance = f1_score_per_instance(predictions_emotions, labels_emotions, list(range(emotion_count)))
    f1scores_emotions_flatten = f1_score_on_flat(flat_pred_emotions, flat_label_emotions, list(range(emotion_count)))
    # f1scores_triggers_instance = f1_score_per_instance(predictions_triggers, labels_triggers, [0,1])
    f1scores_triggers_flatten = f1_score_on_flat(flat_pred_triggers, flat_label_triggers, [0,1])


    return {'accuracy_emotions': round(accuracy_score(flat_pred_emotions, flat_label_emotions), 4),
            'accuracy_triggers': round(accuracy_score(flat_pred_triggers, flat_label_triggers), 4),
            # 'f1scores_emotions_instance': f1scores_emotions_instance,
            'f1scores_emotions_flatten': f1scores_emotions_flatten,
            # 'f1scores_triggers_instance': f1scores_triggers_instance,
            'f1scores_triggers_flatten': f1scores_triggers_flatten}

def print_loss(trainer_history):
    train_loss = []
    val_loss = []
    i = -1
    hist = trainer_history[:-1]
    # Extract val and train loss from the trainer
    for el in hist:
        if i == -1:
            train_loss.append({'loss': el["loss"], "epoch": el["epoch"]})
        else:
            val_loss.append({'loss': el["eval_loss"], "epoch": el["epoch"]})
        i*=-1

    # Data to plot
    train_x = [el["epoch"] for el in train_loss]
    train_y = [el["loss"] for el in train_loss]

    val_x = [el["epoch"] for el in val_loss]
    val_y = [el["loss"] for el in val_loss]

    _, ax = plt.subplots()
    ax.set_xlabel('N. Epoch')
    ax.set_ylabel('Loss')

    ax.plot(train_x, train_y, linewidth=3.0, label="Train")
    ax.plot(val_x, val_y, linestyle='--', linewidth=3.0, label="Validation")
    ax.legend()

    plt.show()

def preprocess_text(tokenizer, dataset, num_emotions):
    my_dict = {
        'episode': [],
        'emotions_id_one_hot_encoding': [],
        'emotions_id': [],
        'triggers': [],
        'dialogue_ids':[],
        'dialogue_mask': [],
        'dialogue_index': [],
        'dialogue_text': [],
        'utterance_ids': [],
        'utterance_mask':[],
        'utterance_index':[]
    }
    all_sentences = [sentence for row in dataset for sentence in row['utterances']]
    concatenated_texts =[" [SEP] ".join(text) for text in dataset['utterances']] 
    all_sentences_tokenized = tokenizer(all_sentences, padding=True, truncation=True)
    concatenated_texts_tokenized = tokenizer(concatenated_texts, padding=True, truncation=True)
    counter = 0
    for i, row in enumerate(dataset):
        for j in range(len(row['utterances'])):
            obj = {
                'episode': row["episode"],
                'dialogue_ids': concatenated_texts_tokenized['input_ids'][i],
                'dialogue_mask': concatenated_texts_tokenized['attention_mask'][i],
                'dialogue_index': i,
                'dialogue_text': concatenated_texts[i],
                'utterance_ids': all_sentences_tokenized['input_ids'][counter],
                'utterance_mask':all_sentences_tokenized['attention_mask'][counter],
                'emotions_id_one_hot_encoding': torch.nn.functional.one_hot(torch.tensor(row["emotions_id"][j]), num_emotions),
                'emotions_id': row["emotions_id"][j],
                'triggers': row['triggers'][j],
                # 'triggers': torch.nn.functional.one_hot(torch.tensor(row['triggers'][j]), 2),
                'utterance_index': j
            }
            counter += 1
            for key in obj.keys():
                my_dict[key].append(obj[key])
    return Dataset.from_dict(my_dict)
    
class DataframeManager():
    def __init__(self, url, dataset_name):
        self.dataset_path = None
        self.raw_df = None
        self.clean_df = None
        self.emotion2id = None
        self.unique_emotions = None

        self.column_triggers = "triggers"
        self.column_utterances = "utterances"
        self.column_emotions = "emotions"
        self.column_emotions_id = "emotions_id"

        self.init_dataset_folder(url, dataset_name)
        self.load_dataframe()
    
    def init_dataset_folder(self, url, dataset_name):
        print(f"Current work directory: {Path.cwd()}")
        dataset_folder = Path.cwd().joinpath("Datasets")

        if not dataset_folder.exists():
            dataset_folder.mkdir(parents=True)

        self.dataset_path = dataset_folder.joinpath(dataset_name)

        if not self.dataset_path.exists():
            print("Downloading dataset into folder...")
            download_url(download_path=self.dataset_path, url=url)
            print("Download complete!")

    def load_dataframe(self):
        try:
            self.raw_df = pd.read_json(self.dataset_path)

        except pd.errors.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
        except FileNotFoundError:
            print(f"File not found: {self.dataset_path}")

    def get_unique_elements_for(self, column: str):
        unique_elements = set()
        for row in self.raw_df[column]:
            for emotion in row:
                unique_elements.add(emotion)
        return unique_elements
    
    def get_id2emotion(self):
        return {i:label for i, label in enumerate(self.unique_emotions)}
    
    def produce_df(self):
        self.clean_df = self.raw_df.copy()

        self.unique_emotions = self.get_unique_elements_for(self.column_emotions)
        self.unique_emotions_len = len(self.unique_emotions)
        # self.unique_emotions_one_hot_encodings = [ [1 if j == i else 0 for j in range(self.unique_emotions_len)] for i in range(self.unique_emotions_len)]

        self.emotion2id = {label:i for i, label in enumerate(self.unique_emotions)}

        # add emotions_id column
        self.clean_df[self.column_emotions_id] = [[self.emotion2id[emotion] for emotion in emotions ] for emotions in self.clean_df[self.column_emotions]]

        # handle None triggers, set to 0 if None
        self.clean_df[self.column_triggers] = self.clean_df[self.column_triggers].apply(lambda x: [int(value) if pd.notna(value) else int(0)  for value in x])  
        # unique_triggers = self.get_unique_elements_for(column_triggers)

        # drop speakers
        self.clean_df = self.clean_df.drop("speakers", axis=1)
        return self.clean_df
    
    def produce_dataset(self, tokenizer, RANDOM_SEED):
        train_df, val_df, test_df = self.split_df(RANDOM_SEED)
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        # train_dataset = Dataset.from_pandas(train_df[0:50])
        # val_dataset = Dataset.from_pandas(val_df[0:10])
        test_dataset = Dataset.from_pandas(test_df)

        train_data_tokenized = preprocess_text(tokenizer, train_dataset, len(self.unique_emotions))
        val_data_tokenized = preprocess_text(tokenizer, val_dataset, len(self.unique_emotions))
        test_data_tokenized = preprocess_text(tokenizer, test_dataset, len(self.unique_emotions))

        train_data_tokenized.set_format("torch")
        val_data_tokenized.set_format("torch")
        test_data_tokenized.set_format("torch")

        return train_data_tokenized, val_data_tokenized, test_data_tokenized

    def split_df(self, seed):
        train_df, val_df, test_df = np.split(self.clean_df.sample(frac=1, random_state=seed), [int(.8*len(self.clean_df)), int(.9*len(self.clean_df))])
        return train_df, val_df, test_df

    def plot_emotion_distribution(self, train_df, val_df, test_df):
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(12, 4)
        def get_arr_of_emotion_counts(df):
            emotions_counts = {label:[] for label in self.unique_emotions}
            for index, row in df.iterrows():
                for index, el in enumerate(row[self.column_emotions]):
                    emotions_counts[el].append(row[self.column_triggers][index])
            return pd.DataFrame({key: pd.value_counts(values)
                                    for key, values in emotions_counts.items()})


        get_arr_of_emotion_counts(train_df).plot(kind='bar', title='Train', ax = axs[0], legend = False)
        get_arr_of_emotion_counts(val_df).plot(kind='bar', title='Validation', ax = axs[1], legend = False)
        get_arr_of_emotion_counts(test_df).plot(kind='bar', title='Test', ax = axs[2], legend = False)

        #test_split.iloc[:, 4:].apply(pd.value_counts).plot(kind='bar', title='Test', ax = axs[2], legend = False)
        fig.legend(labels=list(self.unique_emotions), loc='upper right')
