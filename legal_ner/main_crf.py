import os
import json
import numpy as np
from argparse import ArgumentParser
from nervaluate import Evaluator
from torchcrf import CRF  # Import CRF layer
from transformers import EarlyStoppingCallback
from transformers import AutoModelForTokenClassification, AutoModel
from transformers import Trainer, DefaultDataCollator, TrainingArguments
from torch import nn,cuda
from utils.dataset import LegalNERTokenDataset
import re
import spacy
from peft import LoraConfig, TaskType, get_peft_model
nlp = spacy.load("en_core_web_sm")

class CustomModelWithCRF(nn.Module):
    def __init__(self, model_path, num_labels, freeze=False, hidden_size=768, dropout=0.1, use_lora=False):
        super(CustomModelWithCRF, self).__init__()
        self.device = "cpu" if not cuda.is_available() else "cuda"
        self.encoder = AutoModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
        self.encoder.gradient_checkpointing = True
        self.encoder.gradient_accumulation_steps = acc_step
        if freeze:
            self.encoder.encoder.requires_grad_(False)

        if use_lora:
            model_modules = str(self.encoder.modules)
            pattern = r'\((\w+)\): Linear'
            linear_layer_names = re.findall(pattern, model_modules)

            names = []
            # Print the names of the Linear layers
            for name in linear_layer_names:
                names.append(name)
            target_modules = list(set(names))
            print(f"Found target modules: \n{target_modules}")
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=8, lora_dropout=0.1, bias="all", target_modules=target_modules
            )
            self.encoder = get_peft_model(self.encoder, peft_config)

        # https://github.com/huggingface/transformers/issues/1431
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_out = outputs[0]
        logits = self.linear(self.dropout(sequence_out))
        if labels != None:
            crf_loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction="mean" if batch_size!=1 else "token_mean") # if not mean, it is sum by default
            return (crf_loss, logits)
        else:
            outputs = self.crf.decode(logits, attention_mask.bool())
            return outputs



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

    parser.add_argument(
        "--hidden",
        help="hidden size",
        default=768,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--acc_step",
        help="Gradient accumulation steps",
        default=1,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--use_lora",
        help="Abilitate Peft Lora",
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
    acc_step = args.acc_step
    use_lora = args.use_lora
    
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

        model = CustomModelWithCRF(model_path, num_labels=num_labels, hidden_size=args.hidden, use_lora=use_lora)


        print(model)

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
            gradient_accumulation_steps=acc_step,
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
            logging_steps=50,  # how often to log to W&B
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
        print("**\tCRF ON\t**")
        
        trainer.train()
        trainer.save_model(output_folder)
        trainer.evaluate()

"""
python 3.10
Example of usage:
python main_crf.py \
    --ds_train_path data/NER_TRAIN/NER_TRAIN_ALL.json \
    --ds_valid_path data/NER_DEV/NER_DEV_ALL.json \
    --output_folder results/ \
    --batch 256 \
    --num_epochs 5 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --use_crf
"""