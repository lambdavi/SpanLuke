import os
import json
import numpy as np
from argparse import ArgumentParser
from nervaluate import Evaluator
from torchcrf import CRF  # Import CRF layer
from transformers import EarlyStoppingCallback
from transformers import AutoModelForTokenClassification, AutoModel
from transformers import Trainer, DefaultDataCollator, TrainingArguments
from torch import nn,cuda, zeros_like, bool, where, tensor, BoolTensor, min as m
from utils.dataset import LegalNERTokenDataset


class Primary(nn.Module):
    def __init__(self, model_path, num_labels, freeze=False, hidden_size=768, lstm_hidden_size=256, dropout=0.1, spec_mask=None, weight_ratio=0.2, use_bilstm=False):
        super(Primary, self).__init__()
        self.device = "cpu" if not cuda.is_available() else "cuda"
        self.bert = AutoModel.from_pretrained(model_path, ignore_mismatched_sizes=True, output_hidden_states=use_bilstm)
        if freeze:
            self.bert.encoder.requires_grad_(False)
        # https://github.com/huggingface/transformers/issues/1431
        self.dropout = nn.Dropout(dropout)
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0,
            )
        self.linear = nn.Linear(hidden_size if not self.use_bilstm else lstm_hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.weight_factor = weight_ratio
        self.specialized_labels = spec_mask

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if self.use_bilstm:
            last_hidden_states = outputs.hidden_states[-1]
            sequence_out, _ = self.bilstm(last_hidden_states)
        else:
            sequence_out = outputs[0]
        logits = self.linear(self.dropout(sequence_out))

        logits2 = sec_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_logits_only=True)
        logits2=logits2.to(logits.device)
        # Apply softmax to obtain probabilities
        combined_logits = logits.clone()
        specialized_mask = zeros_like(combined_logits, dtype=bool)
        specialized_mask2 = zeros_like(logits2, dtype=bool)
        for label in self.specialized_labels:
            specialized_mask[:, :, labels_to_idx[label]] = True
            specialized_mask2[:, :, labels_to_idx_sec[label]] = True
    
        combined_logits[specialized_mask] = (1 - self.weight_factor) * logits[specialized_mask] + self.weight_factor * logits2[specialized_mask2]
        
        if labels != None:
            crf_loss = -self.crf(combined_logits, labels, mask=attention_mask.bool(), reduction="mean" if batch_size!=1 else "token_mean") # if not mean, it is sum by default
            return (crf_loss, logits)
        else:
            outputs = self.crf.decode(combined_logits, attention_mask.bool())
            return outputs


class SecondaryModel(nn.Module):
    def __init__(self, model_path, num_labels, freeze=False, hidden_size=768, dropout=0.1):
        super(SecondaryModel, self).__init__()
        self.device = "cpu" if not cuda.is_available() else "cuda"
        self.bert = AutoModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
        self.num_labels = num_labels
        if freeze:
            self.bert.encoder.requires_grad_(False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)
        #self.labels_weights = tensor([0.15, 0.15, 0.15, 0.05, 0.15, 0.15, 0.15, 0.05, 0], device=self.device)
        #self.ce_loss = nn.CrossEntropyLoss(weight=self.labels_weights, reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, return_logits_only=False):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_out = outputs[0]
        logits = self.linear(self.dropout(sequence_out))

        if return_logits_only:
            return sequence_out

        # Compute the loss only for certain labels
        if labels is not None:
            
            # Compute the cross-entropy loss with weights
            # ['O', 'B-ORG', 'B-GPE', 'B-PRECEDENT', 'B-OTHER', 'I-ORG', 'I-GPE', 'I-PRECEDENT', 'I-OTHER']
            loss = self.ce_loss(logits.permute(0, 2, 1), labels)
            
            return (loss, logits)
        else:
            return logits


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
        "--model_path_secondary",
        help="The model path from huggingface/local folder (secondary model)",
        default="roberta-base",
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

    parser.add_argument(
        "--hidden",
        help="Warmup ratio",
        default=768,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--weight_ratio",
        help="Weight ratio of the principal/secondary model",
        default=0.2,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--use_bilstm",
        help="If activated add a bilstm layer in the primary model",
        action="store_true",
        required=False
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
    single_model_path = args.model_path        # e.g. bert-base-uncased
    model_path_secondary = args.model_path_secondary        # e.g. bert-base-uncased
    weight_ratio = args.weight_ratio
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

    original_label_list_second = [
        "ORG",
        "GPE",
        "PRECEDENT",
        "OTHER"
    ]
    labels_list = ["B-" + l for l in original_label_list]
    labels_list += ["I-" + l for l in original_label_list]
    num_labels = len(labels_list) + 1
    print(labels_list)
    labels_list_sec = ["B-" + l for l in original_label_list_second]
    labels_list_sec += ["I-" + l for l in original_label_list_second]
    num_labels_sec = len(labels_list_sec) + 1
    print(labels_list_sec)

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
        model_paths=[single_model_path]
    else:
        model_paths = [
            "nlpaueb/bert-base-uncased-echr", # to delete
            "studio-ousia/luke-large",
            'law-ai/InLegalBERT',
            'microsoft/deberta-v3-base',
            'saibo/legal-roberta-base',
            "geckos/deberta-base-fine-tuned-ner", # not bad, to finetune better
            "studio-ousia/luke-base",
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

        train_ds_small = LegalNERTokenDataset(
            "data/NER_TRAIN/NER_TRAIN_SMALL.json", 
            model_path_secondary, 
            labels_list=labels_list_sec, 
            split="train", 
            use_roberta=use_roberta
        )

        val_ds_small = LegalNERTokenDataset(
            "data/NER_DEV/NER_DEV_SMALL.json", 
            model_path_secondary, 
            labels_list=labels_list_sec, 
            split="val", 
            use_roberta=use_roberta
        )
        labels_to_specialize = ["ORG", "GPE", "PRECEDENT"]
        labels_mask = ["B-" + l for l in labels_to_specialize]
        labels_mask += ["I-" + l for l in labels_to_specialize]
        main_model = Primary(model_path, num_labels=num_labels, hidden_size=args.hidden, spec_mask=labels_mask, weight_ratio=weight_ratio)
        print("MAIN MODEL", main_model, sep="\n")
        sec_model = SecondaryModel(model_path_secondary, num_labels=num_labels_sec, hidden_size=args.hidden)
        print("SECONDARY MODEL", sec_model, sep="\n")
        
        ## Output folder
        new_output_folder = os.path.join(output_folder, 'all')
        new_output_folder = os.path.join(new_output_folder, model_path)
        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)
        ## Training Arguments
        training_args = TrainingArguments(
            output_dir=new_output_folder,
            learning_rate=lr,
            lr_scheduler_type=scheduler_type,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            fp16=False,
            fp16_full_eval=False,
            metric_for_best_model="f1-strict",
            dataloader_num_workers=workers,
            dataloader_pin_memory=True,
            logging_steps=500*batch_size,
            load_best_model_at_end=True,
            report_to="wandb",
            num_train_epochs = 10
        )
        training_args_s = TrainingArguments(
            output_dir=new_output_folder,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            lr_scheduler_type=scheduler_type,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=False,
            fp16_full_eval=False,
            metric_for_best_model="f1-strict",
            dataloader_num_workers=workers,
            dataloader_pin_memory=True,
            report_to="none"
        )

        ## Collator
        data_collator = DefaultDataCollator()

        ## Trainer
        trainer_main = Trainer(
            model=main_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(2)]
        )

        trainer_sec = Trainer(
            model=sec_model,
            args=training_args_s,
            train_dataset=train_ds_small,
            eval_dataset=val_ds_small,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        idx_to_labels = {v[1]: v[0] for v in train_ds_small.labels_to_idx.items()}
        print("**\tCRF ON\t**")
        print("TRAINING AUXILIARY MODEL")
        trainer_sec.train()
        ## Map the labels
        idx_to_labels = {v[1]: v[0] for v in train_ds.labels_to_idx.items()}
        
        labels_to_idx = train_ds.labels_to_idx
        labels_to_idx_sec = train_ds_small.labels_to_idx
        sec_model.eval()
        print(3*"\n")
        print("TRAINING PRIMARY MODEL")
        trainer_main.train()
        trainer_main.save_model(output_folder)
        trainer_main.evaluate()


        
