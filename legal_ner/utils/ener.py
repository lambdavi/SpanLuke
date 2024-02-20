import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.getcwd())
import pandas as pd
from datasets import Dataset
from time import sleep
import json
from transformers import RobertaTokenizerFast, AutoTokenizer

class ENER_DataProcessor():
    def __init__(self, tokenizer=None, data_path="data/ener/all.csv") -> None:
        original = ["BUSINESS", "LOCATION", "PERSON" , "GOVERNMENT", "COURT", "LEGACT", "MISCELLANEOUS"]
        labels_list = ["B-" + l for l in original]
        labels_list += ["I-" + l for l in original]
        self.labels_list = sorted(labels_list + ["O"])[::-1]

        self.labels_to_idx = dict(
            zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
        )

        if tokenizer is not None:
            if "luke" in tokenizer:
                self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) 
        else:
            self.tokenizer=None

        self.data = self.read_data(data_path)

    def label_process(self, example):
        example["tags"] = []
        for i, tag in enumerate(example["ner_tags"]):
            if i == 0:
                prefix = "B"
            else:
                prefix = "B" if example["ner_tags"][i-1] != tag else "I"

            if tag == "O":
                example["tags"].append(self.labels_to_idx[tag])
            else:
                tag = tag.split("-")[1]
                example["tags"].append(self.labels_to_idx[f"{prefix}-{tag}"])
        return example

    def read_data(self, path):
        f = open(path, "r", encoding="utf-8")
        docs = f.read().split("\n,O")
        dataset = []
        print(len(docs))
        for sentence in docs[1:]:# pass -DOCSTART-
            token_list = []
            tag_list = []
            sentence = [line.split(",") for line in sentence.split("\n") if len(line.split(","))==2]
            if len(sentence)!=0:
                for token, tag in sentence:
                    token_list.append(token)
                    tag_list.append(tag)
                
                dataset.append({"tokens": token_list, "ner_tags": tag_list})
        
        dataset = pd.DataFrame(dataset, columns=["tokens", "ner_tags"])
        dataset = Dataset.from_pandas(dataset)
        return dataset

    # to jsonl
    def to_jsonl(ener):
        with open(f"ener/ener.jsonl", "w") as f:
            for data in ener:
                print(data) 
                data.pop("ner_tags", None)
                data["ner_tags"] = data.pop("tags")
                f.write(json.dumps(data)+'\n')


    def tokenize_and_align_labels(self, examples, label_all_tokens=True): 
        print(examples["tokens"], examples["ner_tags"])
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) 
        labels = [] 
        for i, label in enumerate(examples["ner_tags"]): 
            word_ids = tokenized_inputs.word_ids(batch_index=i) 
            # word_ids() => Return a list mapping the tokens
            # to their actual word in the initial sentence.
            # It Returns a list indicating the word corresponding to each token. 
            previous_word_idx = None 
            label_ids = []
            # Special tokens like `` and `<\s>` are originally mapped to None 
            # We need to set the label to -100 so they are automatically ignored in the loss function.
            for word_idx in word_ids: 
                if word_idx is None: 
                    # set –100 as the label for these special tokens
                    label_ids.append(-100)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                elif word_idx != previous_word_idx:
                    # if current word_idx is != prev then its the most regular case
                    # and add the corresponding token     
                    label_ids.append(label[word_idx]) 
                else: 
                    # to take care of sub-words which have the same word_idx
                    # set -100 as well for them, but only if label_all_tokens == False
                    label_ids.append(-100) 
                    # mask the subword representations after the first subword
                    
                previous_word_idx = word_idx 
            labels.append(label_ids) 
        tokenized_inputs["labels"] = labels 
        print(tokenized_inputs["labels"])
        return tokenized_inputs 
    
    def get_ener_dataset(self):
        ener = self.data.map(self.label_process)
        ener = ener.remove_columns("ner_tags")
        ener = ener.rename_column("tags", "ner_tags")
        if self.tokenizer:
            ener = ener.select(range(10)).map(self.tokenize_and_align_labels, batched=True)
        return ener.train_test_split(0.2, seed=42)
