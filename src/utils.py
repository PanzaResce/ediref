import numpy as np
import pandas as pd
import torch
import urllib
import tqdm
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot
from pathlib import Path
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

def compute_metrics(eval_pred, id2emotion):
    predictions_emotions, labels_emotions, predictions_triggers, labels_triggers = eval_pred
    flat_pred_emotions = [item for list_of_pred in predictions_emotions for item in list_of_pred]
    flat_label_emotions = [item for list_of_pred in labels_emotions for item in list_of_pred]
    flat_pred_triggers = [item for list_of_pred in predictions_triggers for item in list_of_pred]
    flat_label_triggers = [item for list_of_pred in labels_triggers for item in list_of_pred]

    f1scores_emotions_instance = f1_score_per_instance(predictions_emotions, labels_emotions, list(id2emotion.keys()))
    f1scores_emotions_flatten = f1_score_on_flat(flat_pred_emotions, flat_label_emotions, list(id2emotion.keys()))
    f1scores_triggers_instance = f1_score_per_instance(predictions_triggers, labels_triggers, [0,1])
    f1scores_triggers_flatten = f1_score_on_flat(flat_pred_triggers, flat_label_triggers, [0,1])


    return {'accuracy_emotions': round(accuracy_score(flat_pred_emotions, flat_label_emotions), 4),
            'accuracy_triggers': round(accuracy_score(flat_pred_triggers, flat_label_triggers), 4),
            'f1scores_emotions_instance': f1scores_emotions_instance,
            'f1scores_emotions_flatten': f1scores_emotions_flatten,
            'f1scores_triggers_instance': f1scores_triggers_instance,
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

def preprocess_text(tokenizer, rows):
    tokenized_texts = []
    for text in rows['utterances']:
        tokenized_sentences = tokenizer(text, padding=True, truncation=True)
        tokenized_texts.append(tokenized_sentences)

    # Create a dictionary for the dataset
    dataset_dict = {
        'input_ids': [text_tokens['input_ids'] for text_tokens in tokenized_texts],
        'attention_mask': [text_tokens['attention_mask'] for text_tokens in tokenized_texts],
    }
    return dataset_dict

def get_datasets(train, val, test, tokenizer):
    train_data = train.map(lambda rows: preprocess_text(tokenizer, rows), batched=True)
    val_data = val.map(lambda rows: preprocess_text(tokenizer, rows), batched=True)
    test_data = test.map(lambda rows: preprocess_text(tokenizer, rows), batched=True)

    train_data.set_format("torch")
    val_data.set_format("torch")
    test_data.set_format("torch")

    return train_data, val_data, test_data

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

    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])

    def produce_df(self):
        self.clean_df = self.raw_df.copy()

        self.unique_emotions = self.get_unique_elements_for(self.column_emotions)

        self.emotion2id = {label:i for i, label in enumerate(self.unique_emotions)}
        # id2emotion = {i:label for i, label in enumerate(unique_emotions)}

        # add emotions_id column as one-hot encodings
        self.clean_df[self.column_emotions_id] = [self.get_one_hot(np.array([self.emotion2id[emotion] for emotion in emotions ]), 7) for emotions in self.clean_df["emotions"]]

        # handle None triggers, set to 0 if None
        self.clean_df[self.column_triggers] = self.clean_df[self.column_triggers].apply(lambda x: [int(value) if pd.notna(value) else int(0)  for value in x])  
        # unique_triggers = self.get_unique_elements_for(column_triggers)

        # drop speakers
        self.clean_df = self.clean_df.drop("speakers", axis=1)
        return self.clean_df
    
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
