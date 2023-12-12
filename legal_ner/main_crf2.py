import os
import json
import numpy as np
from argparse import ArgumentParser
from nervaluate import Evaluator
from transformers import EarlyStoppingCallback
from transformers import AutoModelForTokenClassification, AutoModel
from transformers import Trainer, DefaultDataCollator, TrainingArguments
from torch import nn
from utils.dataset import LegalNERTokenDataset
import torch
import spacy
nlp = spacy.load("en_core_web_sm")

## Define the model with CRF layer
class CustomModelWithCRF(AutoModelForTokenClassification):
    def __init__(self, model_path, num_labels, config):
        super().__init__(config)
        self.num_labels = num_labels
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bert = AutoModel(model_path)
        self.dropout = nn.Dropout(0.2)
        self.hidden2label = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.use_crf=True

        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        self.end_transitions = nn.Parameter(torch.empty(num_labels))
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))

        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _compute_log_denominator(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len = features.shape[0]

        log_score_over_all_seq = self.start_transitions + features[0]

        for i in range(1, seq_len):
            next_log_score_over_all_seq = torch.logsumexp(
                log_score_over_all_seq.unsqueeze(2) + self.transitions + features[i].unsqueeze(1),
                dim=1,
            )
            log_score_over_all_seq = torch.where(
                mask[i].unsqueeze(1),
                next_log_score_over_all_seq,
                log_score_over_all_seq,
            )
        log_score_over_all_seq += self.end_transitions
        return torch.logsumexp(log_score_over_all_seq, dim=1)

    def _compute_log_numerator(self, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, bs, _ = features.shape

        score_over_seq = self.start_transitions[labels[0]] + features[0, torch.arange(bs), labels[0]]

        for i in range(1, seq_len):
            score_over_seq += (
                self.transitions[labels[i - 1], labels[i]] + features[i, torch.arange(bs), labels[i]]
            ) * mask[i]
        seq_lens = mask.sum(dim=0) - 1
        last_tags = labels[seq_lens.long(), torch.arange(bs)]
        score_over_seq += self.end_transitions[last_tags]
        return score_over_seq

    def get_bert_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        hidden = self.bert(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        hidden = self.dropout(hidden)
        return self.hidden2label(hidden), hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        features, _ = self.get_bert_features(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.bool()

        if self.use_crf:
            print("Performing crf..")
            features = torch.swapaxes(features, 0, 1)
            attention_mask = torch.swapaxes(attention_mask, 0, 1)
            labels = torch.swapaxes(labels, 0, 1)

            log_numerator = self._compute_log_numerator(features=features, labels=labels, mask=attention_mask)
            log_denominator = self._compute_log_denominator(features=features, mask=attention_mask)

            return torch.mean(log_denominator - log_numerator)
        else:
            return self.cross_entropy(
                features.flatten(end_dim=1),
                torch.where(attention_mask.bool(), labels, -100).flatten(end_dim=1),
            )

    def _viterbi_decode(self, features: torch.Tensor, mask: torch.Tensor):
        seq_len, bs, _ = features.shape

        log_score_over_all_seq = self.start_transitions + features[0]

        backpointers = torch.empty_like(features)

        for i in range(1, seq_len):
            next_log_score_over_all_seq = (
                log_score_over_all_seq.unsqueeze(2) + self.transitions + features[i].unsqueeze(1)
            )

            next_log_score_over_all_seq, indices = next_log_score_over_all_seq.max(dim=1)

            log_score_over_all_seq = torch.where(
                mask[i].unsqueeze(1),
                next_log_score_over_all_seq,
                log_score_over_all_seq,
            )
            backpointers[i] = indices

        backpointers = backpointers[1:].int()

        log_score_over_all_seq += self.end_transitions
        seq_lens = mask.sum(dim=0) - 1

        best_paths = []
        for seq_ind in range(bs):
            best_label_id = torch.argmax(log_score_over_all_seq[seq_ind]).item()
            best_path = [best_label_id]

            for backpointer in reversed(backpointers[: seq_lens[seq_ind]]):
                best_path.append(backpointer[seq_ind][best_path[-1]].item())

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        features, _ = self.get_bert_features(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.bool()

        if self.use_crf:
            features = torch.swapaxes(features, 0, 1)
            mask = torch.swapaxes(attention_mask, 0, 1)
            return self._viterbi_decode(features=features, mask=mask)
        else:
            labels = torch.argmax(features, dim=2)
            predictions = []
            for i in range(len(labels)):
                predictions.append(labels[i][attention_mask[i]].tolist())
            return predictions
        

############################################################
#                                                          #
#                           MAIN                           #
#                                                          #
############################################################ 
if __name__ == "__main__":

    parser = ArgumentParser(description="Training of LUKE model")
    parser.add_argument(
        "--ds_train_path",
        help="Path of train dataset file",
        default="data/NER_TRAIN/NER_TRAIN_ALL.json",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--ds_valid_path",
        help="Path of validation dataset file",
        default="data/NER_DEV/NER_DEV_ALL.json",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder",
        default="results/",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--model_path",
        help="The model path from huggingface/local folder",
        default=None,
        required=False,
        type=str,
    )
    parser.add_argument(
        "--scheduler",
        help="Scheduler type among: linear, polynomial, reduce_lr_on_plateau, cosine, constant",
        choices=["linear", "polynomial", "reduce_lr_on_plateau", "cosine", "constant"],
        default="linear",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--batch",
        help="Batch size",
        default=1,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--workers",
        help="Number of workers",
        default=4,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of training epochs",
        default=5,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        default=1e-5,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        help="Weight decay",
        default=0.01,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--warmup_ratio",
        help="Warmup ratio",
        default=0.06,
        required=False,
        type=float,
    )

    args = parser.parse_args()

    ## Parameters
    ds_train_path = args.ds_train_path  # e.g., 'data/NER_TRAIN/NER_TRAIN_ALL.json'
    ds_valid_path = args.ds_valid_path  # e.g., 'data/NER_DEV/NER_DEV_ALL.json'
    output_folder = args.output_folder  # e.g., 'results/'
    batch_size = args.batch             # e.g., 256 for luke-based, 1 for bert-based
    num_epochs = args.num_epochs        # e.g., 5
    lr = args.lr                        # e.g., 1e-4 for luke-based, 1e-5 for bert-based
    weight_decay = args.weight_decay    # e.g., 0.01
    warmup_ratio = args.warmup_ratio    # e.g., 0.06
    workers = args.workers              # e.g., 4
    scheduler_type = args.scheduler     # e.g., linear

    ## Define the labels
    original_label_list = [
        "COURT",
        "PETITIONER",
        "RESPONDENT",
        "JUDGE",
        "DATE",
        "ORG",
        "GPE",
        "STATUTE",
        "PROVISION",
        "PRECEDENT",
        "CASE_NUMBER",
        "WITNESS",
        "OTHER_PERSON",
        "LAWYER"
    ]
    labels_list = ["B-" + l for l in original_label_list]
    labels_list += ["I-" + l for l in original_label_list]
    num_labels = len(labels_list) + 1

    ## Compute metrics
    def compute_metrics(pred):

        # Preds
        predictions = np.argmax(pred.predictions, axis=-1)
        predictions = np.concatenate(predictions, axis=0)
        prediction_ids = [[idx_to_labels[p] if p != -100 else "O" for p in predictions]]

        # Labels
        labels = pred.label_ids
        labels = np.concatenate(labels, axis=0)
        labels_ids = [[idx_to_labels[p] if p != -100 else "O" for p in labels]]
        unique_labels = list(set([l.split("-")[-1] for l in list(set(labels_ids[0]))]))
        unique_labels.remove("O")

        # Evaluator
        evaluator = Evaluator(
            labels_ids, prediction_ids, tags=unique_labels, loader="list"
        )
        results, results_per_tag = evaluator.evaluate()
        print("")
        for k,v in results_per_tag.items():
            print(f"{k}: {v['ent_type']['f1']}")
        return {
            "f1-type-match": 2
            * results["ent_type"]["precision"]
            * results["ent_type"]["recall"]
            / (results["ent_type"]["precision"] + results["ent_type"]["recall"] + 1e-9),
            "f1-partial": 2
            * results["partial"]["precision"]
            * results["partial"]["recall"]
            / (results["partial"]["precision"] + results["partial"]["recall"] + 1e-9),
            "f1-strict": 2
            * results["strict"]["precision"]
            * results["strict"]["recall"]
            / (results["strict"]["precision"] + results["strict"]["recall"] + 1e-9),
            "f1-exact": 2
            * results["exact"]["precision"]
            * results["exact"]["recall"]
            / (results["exact"]["precision"] + results["exact"]["recall"] + 1e-9),
        }

    if args.model_path:
        model_paths=[args.model_path]
    else:
        model_paths = [
            "bert-base-cased"
        ]
    for model_path in model_paths:

        print("MODEL: ", model_path)

        ## Define the train and test datasets
        use_roberta = False
        if "luke" in model_path or "roberta" in model_path or "berta" in model_path or "xlm" in model_path or "span" in model_path or "distilbert" in model_path:
            use_roberta = True

        train_ds = LegalNERTokenDataset(
            ds_train_path, 
            model_path, 
            labels_list=labels_list, 
            split="train", 
            use_roberta=use_roberta
        )

        val_ds = LegalNERTokenDataset(
            ds_valid_path, 
            model_path, 
            labels_list=labels_list, 
            split="val", 
            use_roberta=use_roberta
        )

        
        model = CustomModelWithCRF.from_pretrained(
            model_path, 
            num_labels=num_labels, 
            ignore_mismatched_sizes=True
        )

        ## Map the labels
        idx_to_labels = {v[1]: v[0] for v in train_ds.labels_to_idx.items()}

        ## Output folder
        new_output_folder = os.path.join(output_folder, 'all')
        new_output_folder = os.path.join(new_output_folder, model_path)
        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)

        ## Training Arguments
        training_args = TrainingArguments(
            output_dir=new_output_folder,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            lr_scheduler_type=scheduler_type,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True if ("span" not in model_path and "albert" not in model_path) else False,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
            fp16=False,
            fp16_full_eval=False,
            metric_for_best_model="f1-strict",
            dataloader_num_workers=workers,
            dataloader_pin_memory=True,
            report_to="wandb",
            logging_steps=50 if ("bert-" not in model_path and "albert" not in model_path) else 3000,  # how often to log to W&B
        )

        ## Collator
        data_collator = DefaultDataCollator()

        ## Trainer

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(2)]
        )

        ## Train the model and save it
        print("** CRF ON - STARTING **")
        trainer.train()
        trainer.save_model(output_folder)
        trainer.evaluate()



"""python 3.10
Example of usage:
python main_crf.py \
    --ds_train_path data/NER_TRAIN/NER_TRAIN_ALL.json \
    --ds_valid_path data/NER_DEV/NER_DEV_ALL.json \
    --output_folder results/ \
    --batch 256 \
    --num_epochs 5 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06
"""