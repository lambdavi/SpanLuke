import torch
import json
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, RobertaTokenizerFast
from datasets import Dataset, DatasetDict
from utils.utils import match_labels
import spacy
nlp = spacy.load("en_core_web_sm")

############################################################
#                                                          #
#                      DATASET CLASS                       #
#                                                          #
############################################################ 

class LegalNERTokenDataset(Dataset):
    def __init__(self, dataset_path, model_path, labels_list=None, split="train", use_roberta=False):
        self.model_path = model_path
        self._data = json.load(open(dataset_path))
        self.split = split
        self.use_roberta = use_roberta
        print("Using roberta config" if use_roberta else "Not using Roberta config")
        self.column_names = ["tokens", "ner_tags"]
        if self.use_roberta:     ## Load the right tokenizer
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path) 
        self.labels_list = sorted(labels_list + ["O"])[::-1]
        if self.labels_list is not None:
            self.labels_to_idx = dict(
                zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
            )
        print(self.labels_list)

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
        #print(self.use_roberta)
        if inputs.get("token_type_ids") != None:
            inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0).long()

        if "span" in self.model_path:
            inputs["tokens"] = self.tokenizer.decode(inputs["input_ids"])
        ## Get the labels
        if self.labels_list:
            labels = torch.tensor(aligned_labels).squeeze(-1).long()
            column_name = "labels" if "span" not in self.model_path else "ner_tags"
            if labels.shape[0] < inputs["attention_mask"].shape[0]:
                pad_x = torch.zeros((inputs["input_ids"].shape[0],))
                pad_x[: labels.size(0)] = labels
                inputs[column_name] = aligned_labels
            else:
                inputs[column_name] = labels[: inputs["attention_mask"].shape[0]]
        return inputs

## FOR SPAN
def load_legal_ner(train_data_folder: str):
    pass