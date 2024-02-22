import torch
import json
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, RobertaTokenizerFast
from datasets import DatasetDict, Dataset as DatasetHF
from utils.utils import match_labels

import spacy
nlp = spacy.load("en_core_web_sm")

############################################################
#                                                          #
#                     LEGAL DATASET CLASS                  #
#                                                          #
############################################################ 
class LegalNERTokenDataset(Dataset):
    
    def __init__(self, dataset_path, model_path, labels_list=None, split="train", use_roberta=False):
        self.data = json.load(open(dataset_path))
        self.split = split
        self.use_roberta = use_roberta
        if self.use_roberta:     ## Load the right tokenizer
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path) 
        self.labels_list = sorted(labels_list + ["O"])[::-1]

        if self.labels_list is not None:
            self.labels_to_idx = dict(
                zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["data"]["text"]

        ## Get the annotations
        annotations = [
            {
                "start": v["value"]["start"],
                "end": v["value"]["end"],
                "labels": v["value"]["labels"][0],
            }
            for v in item["annotations"][0]["result"]
        ]

        ## Tokenize the text
        if not self.use_roberta:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                verbose=False
                )
        else:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                verbose=False, 
                padding='max_length'
            )

        ## Match the labels
        aligned_labels = match_labels(inputs, annotations)
        aligned_labels = [self.labels_to_idx[l] for l in aligned_labels]
        inputs["input_ids"] = inputs["input_ids"].squeeze(0).long()
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0).long()
        if not self.use_roberta:
            inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0).long()

        ## Get the labels
        if self.labels_list:
            labels = torch.tensor(aligned_labels).squeeze(-1).long()

            if labels.shape[0] < inputs["attention_mask"].shape[0]:
                pad_x = torch.zeros((inputs["input_ids"].shape[0],))
                pad_x[: labels.size(0)] = labels
                inputs["labels"] = aligned_labels
            else:
                inputs["labels"] = labels[: inputs["attention_mask"].shape[0]]

        return inputs

  

## FOR SPAN - LEGAL NER
def load_legal_ner(train_data_folder: str, dev_data_folder: str):
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

############################################################
#                                                          #
#                     E-NER DATASET CLASS                  #
#                                                          #
############################################################ 
class ENER_Dataset():
    def __init__(self, train_ds_path, test_ds_path, labels_list, tokenizer=None) -> None:

        self.labels_list = sorted(labels_list + ["O"])[::-1]

        self.labels_to_idx = dict(
            zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
        )

        if isinstance(tokenizer, str):
            if "luke" in tokenizer:
                print("Using roberta as tokenizer..")
                self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
            else:
                print(f"Using {tokenizer} as tokenizer..")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) 
        else:
            print("Tokenizer already set")
            self.tokenizer=tokenizer

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
        ret["test"] = DatasetHF.from_list(data)

        return DatasetDict(ret)

    # to jsonl
    def to_jsonl(self, ener):
        with open(f"ener/ener.jsonl", "w") as f:
            for data in ener:
                f.write(json.dumps(data)+'\n')

    def tokenize_and_align_labels(self, examples):

        tokenized_inputs = self.tokenizer(examples, truncation=True)

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
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def get_ener_dataset(self):
        ener = self.data
        ener = ener.map(self.tokenize_and_align_labels, batched=True)
        return ener
