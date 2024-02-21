import torch
import json
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, RobertaTokenizerFast
from datasets import DatasetDict, Dataset as DatasetHF
from utils.utils import match_labels, match_labels_ner
import pandas as pd


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

class ENERTokenDataset(Dataset):
    def __init__(self, dataset_path, model_path, labels_list=None, split="train", use_roberta=False):
        self.data = self._read_data("data/ener/all2.csv")
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

    def _label_process(self, example):
        example["tags"] = []
        for i, tag in enumerate(example["ner_tags"]):
            if i == 0:
                prefix = "B"
            else:
                prefix = "B" if example["ner_tags"][i-1] != tag else "I"

            if tag == "O":
                example["tags"].append(self.labels_to_idx[tag])
            else:
                example["tags"].append(self.labels_to_idx[f"{prefix}-{tag}"])
        return example
    
    def _read_data(self, path):
        f = open(path, "r", encoding="utf-8")
        docs = f.read().split("\n,O")
        dataset = []
        for sentence in docs[1:]:# pass -DOCSTART-
            token_list = []
            tag_list = []
            sentence = [line.split(",") for line in sentence.split("\n") if len(line.split(","))==2]
            if len(sentence)!=0:
                for token, tag in sentence:
                    token_list.append(token)
                    tag_list.append(tag)
                
                dataset.append({"tokens": token_list, "ner_tags": tag_list})
        return dataset
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self._label_process(self.data[idx])      

        ## Tokenize the text
        if not self.use_roberta:
            inputs = self.tokenizer(
                example["tokens"], 
                return_tensors="pt", 
                truncation=True, 
                verbose=False,
                is_split_into_words=True
            )
        else:
            inputs = self.tokenizer(
                example["tokens"], 
                return_tensors="pt", 
                truncation=True, 
                verbose=False, 
                padding='max_length',
                is_split_into_words=True
            )

        ## Match the labels
        aligned_labels = match_labels_ner(inputs, example)
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
  

## FOR SPAN
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