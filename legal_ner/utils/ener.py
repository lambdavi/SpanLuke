import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.getcwd())
import pandas as pd
from datasets import Dataset
from time import sleep
import json
original = ["BUSINESS", "LOCATION", "PERSON" , "GOVERNMENT", "COURT", "LEGISLATION/ACT", "MISCELLANEOUS"]
entities = ["B-" + l for l in original]
entities += ["I-" + l for l in original]
#print(entities)
entity_to_tag = {e: i+1 for i, e in enumerate(sorted(entities))}
entity_to_tag["O"]=0
#print(entity_to_tag)

def label_process(example):
    example["tags"] = []
    for i, tag in enumerate(example["ner_tags"]):
        if i == 0:
            prefix = "B"
        else:
            prefix = "B" if example["ner_tags"][i-1] != tag else "I"

        if tag == "O":
            example["tags"].append(entity_to_tag[tag])
        else:
            tag = tag.split("-")[1]
            example["tags"].append(entity_to_tag[f"{prefix}-{tag}"])
    return example

def read_data(path):
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

def get_ener_dataset():
    ener = read_data("data/ener/all.csv").map(label_process)
    ener = ener.remove_columns("ner_tags")
    ener = ener.rename_column("tags", "ner_tags")
    return ener.train_test_split(0.15)
