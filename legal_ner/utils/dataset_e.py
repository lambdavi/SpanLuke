import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, file_path, labels_list, tokenizer_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sentences, self.labels = self.read_data(file_path)
        self.labels_list = sorted(labels_list + ["O"])[::-1]
        if self.labels_list is not None:
            self.label_to_id = dict(
                zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
            )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = [self.label_to_id[l] for l in self.labels[idx]]
        print(sentence, labels)
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=10,  # adjust as needed
            return_tensors='pt',
            return_attention_mask=True,
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(labels)}

    def read_data(self, file_path):
        sentences = []
        labels = []

        import pandas as pd

        df = pd.read_csv(file_path, header=None, sep=" ")
        print(df.head(30))
        current_sentence = []
        current_labels = []
        for i, row in df.iterrows():
            word = str(row[0])
            label = str(row[1])

            # Convert I-* labels to B-* if it's the first token in a sequence
            if label.startswith("I-"):
                if not current_labels or current_labels[-1][2:] != label[2:]:
                    label = "B-" + label[2:]

            current_sentence.append(word)
            current_labels.append(label)

            # Assume each row in the CSV corresponds to a sentence
            if i == len(df) - 1 or pd.isna(df.loc[i + 1, 0]):
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence = []
                current_labels = []

        return sentences, labels

# Example usage
file_path = "legal_ner/data/edgar_all_4.csv"
original_label_list = ["PER", "ORG", "LOC", "MISC"]
labels_list = ["B-" + l for l in original_label_list]
labels_list += ["I-" + l for l in original_label_list]
custom_dataset = CustomDataset(file_path, labels_list)

# Example of accessing a sample from the dataset
sample = custom_dataset[0]
print("Input IDs:", sample["input_ids"])
print("Attention Mask:", sample["attention_mask"])
print("Label IDs:", sample["labels"])


