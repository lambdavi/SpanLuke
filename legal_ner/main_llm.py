import os
import json
import numpy as np
from argparse import ArgumentParser
from nervaluate import Evaluator
from span_marker import SpanMarkerModel, Trainer as SpanTrainer
from transformers import EarlyStoppingCallback
from transformers import AutoModelForTokenClassification, RobertaTokenizerFast
from transformers import Trainer, DefaultDataCollator, TrainingArguments, DataCollatorForTokenClassification
from sklearn.metrics import confusion_matrix
from utils.dataset import LegalNERTokenDataset
from modeling_llama import (
    LlamaForSequenceClassification, LlamaForTokenClassification,
    UnmaskingLlamaForSequenceClassification, UnmaskingLlamaForTokenClassification,
)
import spacy
nlp = spacy.load("en_core_web_sm")


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
        "--acc_step",
        help="Gradient accumulation steps",
        default=1,
        required=False,
        type=int,
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
    acc_step = args.acc_step

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

        # Confusion Matrix
        cm = confusion_matrix(labels_ids[0], prediction_ids[0], labels=unique_labels)

        # Print Confusion Matrix
        print("Confusion Matrix:")
        print(cm)
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
            'TinyPixel/Llama-2-7B-bf16-sharded'
        ]
    for model_path in model_paths:

        print("MODEL: ", model_path)

        ## Define the train and test datasets
        use_roberta = False
        if "luke" in model_path or "roberta" in model_path or "llama" in model_path or "berta" in model_path or "xlm" in model_path or "span" in model_path or "distilbert" in model_path:
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

        ## Define the model
        if "span" in model_path:
            # Download from the 🤗 Hub

            encoder_id = "roberta-base"
            model = SpanMarkerModel.from_pretrained(
                # Required arguments
                encoder_id,
                labels=['O']+original_label_list,
                # Optional arguments
                model_max_length=256,
                entity_max_length=8,
                ignore_mismatched_sizes=True
                # To improve the generated model card
            )
            """model = SpanMarkerModel.from_pretrained(
                model_path,
                num_labels=num_labels,
                labels=original_label_list,
                ignore_mismatched_sizes=True
            )"""
        else:
            # Run inference
            model = LlamaForTokenClassification.from_pretrained(
                model_path, 
                num_labels=num_labels, 
                ignore_mismatched_sizes=True
            ).bfloat16()
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
            gradient_checkpointing=True if "span" not in model_path else False,
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
            logging_steps=50 if "bert-" not in model_path else 3000,  # how often to log to W&B
        )

        ## Collator
        data_collator = DataCollatorForTokenClassification(tokenizer=RobertaTokenizerFast.from_pretrained("roberta-base"))

        ## Trainer
        if "span" not in model_path:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(2)]
            )
        else:
            trainer = SpanTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            )

        ## Train the model and save it
        trainer.train()
        trainer.save_model(output_folder)
        trainer.evaluate()

    


"""python 3.10
Example of usage:
python main.py \
    --ds_train_path data/NER_TRAIN/NER_TRAIN_ALL.json \
    --ds_valid_path data/NER_DEV/NER_DEV_ALL.json \
    --output_folder results/ \
    --batch 256 \
    --num_epochs 5 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06
"""