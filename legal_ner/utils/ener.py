import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.getcwd())
import pandas as pd
from datasets import Dataset
import json
from transformers import RobertaTokenizerFast, AutoTokenizer
from datasets import DatasetDict, Dataset as DatasetHF

class ENER_DataProcessor():
    def __init__(self, train_ds_path, test_ds_path, tokenizer=None) -> None:
        original = ["BUSINESS", "LOCATION", "PERSON" , "GOVERNMENT", "COURT", "LEGACT", "MISCELLANEOUS"]
        labels_list = ["B-" + l for l in original]
        labels_list += ["I-" + l for l in original]
        self.labels_list = sorted(labels_list + ["O"])[::-1]

        self.labels_to_idx = dict(
            zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
        )

        if tokenizer is not None:
            if "luke" in tokenizer:
                print("Using roberta as tokenizer..")
                self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
            else:
                print(f"Using {tokenizer} as tokenizer..")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) 
        else:
            self.tokenizer=None

        self.data = self.read_data(train_ds_path, test_ds_path)

    def read_data(self, train_data_folder: str, dev_data_folder: str):
        ret = {}
        
        # TRAIN
        data = []
        with open(f"{train_data_folder}l", 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret["train"] = DatasetHF.from_list(data)

        data = []

        with open(f"{dev_data_folder}l", 'r') as reader:
            for line in reader:
                data.append(json.loads(line))
        ret["dev"] = DatasetHF.from_list(data)

        return DatasetDict(ret)

    # to jsonl
    def to_jsonl(self, ener):
        with open(f"ener/ener.jsonl", "w") as f:
            for data in ener:
                f.write(json.dumps(data)+'\n')

    def tokenize_and_align_labels(self, examples, label_all_tokens=True):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def get_ener_dataset(self):
        ener = self.data
        if self.tokenizer:
            ener = ener.map(self.tokenize_and_align_labels, batched=True)
        return ener
