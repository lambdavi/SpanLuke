import torch
import json
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, RobertaTokenizerFast
from utils.utils import match_labels

import spacy
nlp = spacy.load("en_core_web_sm")

############################################################
#                                                          #
#                      DATASET CLASS                       #
#                                                          #
############################################################ 
# TODO: Add support for the SpanMarker model, needed columns: --
class LegalNERTokenDataset(Dataset):
    def __init__(self, dataset_path, model_path, labels_list_small, labels_list=None, split="train", use_roberta=False):
        self.model_path = model_path
        self.data = json.load(open(dataset_path))
        self.split = split
        self.use_roberta = use_roberta
        print("Using roberta config" if use_roberta else "Not using Roberta config")
        self.column_names = ["tokens", "ner_tags"]
        if self.use_roberta:     ## Load the right tokenizer
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.l2l={}
        for l in labels_list:
            self.l2l[l]=l if l in labels_list_small else l[:2]+"OTHER"
        print("L2L: ", self.l2l)
        tmp = set([l if l in labels_list_small else l[:2]+"OTHER" for l in labels_list])
        print("TMP: ",tmp)
        self.labels_list_small = sorted([tmp]+ ["O"])[::-1]
        self.labels_list = sorted(labels_list + ["O"])[::-1]

        if self.labels_list is not None:
            self.labels_to_idx = dict(
                zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
            )
        
        if self.labels_list_small is not None:
            self.labels_to_idx_s = dict(
                zip(sorted(self.labels_list_small)[::-1], range(len(self.labels_list_small)))
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
        aligned_labels_small = [self.labels_to_idx_s[l] for l in aligned_labels]
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
            labels_s = torch.tensor(aligned_labels_small).squeeze(-1).long()
            if labels.shape[0] < inputs["attention_mask"].shape[0]:
                pad_x = torch.zeros((inputs["input_ids"].shape[0],))
                pad_x[: labels.size(0)] = labels
                inputs["labels"] = aligned_labels
                inputs["labels_s"] = aligned_labels_small
            else:
                inputs["labels"] = labels[: inputs["attention_mask"].shape[0]]
                inputs["labels_s"] = labels_s[: inputs["attention_mask"].shape[0]]
        return inputs