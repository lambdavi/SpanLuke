import os
import json
import numpy as np
from argparse import ArgumentParser
from nervaluate import Evaluator
from transformers import EarlyStoppingCallback
from transformers import AutoModelForTokenClassification
from transformers import Trainer, DefaultDataCollator, TrainingArguments
from utils.dataset import LegalNERTokenDataset
from transformers import TrainingArguments
from span_marker import SpanMarkerModel, SpanMarkerModelCardData, Trainer
import json
from span_marker import SpanMarkerModel, Trainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import torch
from span_marker.tokenizer import SpanMarkerTokenizer
from span_marker.configuration import SpanMarkerConfig
from nervaluate import Evaluator

############################################################
#                                                          #
#                           MAIN                           #
#                                                          #
############################################################ 
if __name__ == "__main__":

    parser = ArgumentParser(description="Training of LUKE model")
    parser.add_argument(
        "--ds_path",
        help="Path of data folder",
        default="data",
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
    ds_path = args.ds_train_path  # e.g., 'data/NER_TRAIN/NER_TRAIN_ALL.json'
    output_folder = args.output_folder  # e.g., 'results/'
    batch_size = args.batch             # e.g., 256 for luke-based, 1 for bert-based
    num_epochs = args.num_epochs        # e.g., 5
    lr = args.lr                        # e.g., 1e-4 for luke-based, 1e-5 for bert-based
    weight_decay = args.weight_decay    # e.g., 0.01
    warmup_ratio = args.warmup_ratio    # e.g., 0.06
    workers = args.workers              # e.g., 4
    scheduler_type = args.scheduler     # e.g., linear
    acc_step = args.acc_step
    model_path = args.model_path

    print("MODEL: ", args.model_path)

    ## Define the train and test datasets
    use_roberta = False
    if "luke" in model_path or "roberta" in model_path or "berta" in model_path or "xlm" in model_path or "span" in model_path or "distilbert" in model_path:
        use_roberta = True

    dataset = load_legal_ner(ds_path)

    entities = ['B-COURT', 'B-PETITIONER', 'B-RESPONDENT', 'B-JUDGE', 'B-DATE', 'B-ORG', 'B-GPE', 'B-STATUTE', 'B-PROVISION', 'B-PRECEDENT', 'B-CASE_NUMBER', 'B-WITNESS', 'B-OTHER_PERSON', 'B-LAWYER', 'I-COURT', 'I-PETITIONER', 'I-RESPONDENT', 'I-JUDGE', 'I-DATE', 'I-ORG', 'I-GPE', 'I-STATUTE', 'I-PROVISION', 'I-PRECEDENT', 'I-CASE_NUMBER', 'I-WITNESS', 'I-OTHER_PERSON', 'I-LAWYER']
    labels = ["O"]+entities


    # Initialize a SpanMarkerModel using an encoder, e.g. BERT, and the labels:
    #encoder_id = "xlm-roberta-base"
    model = SpanMarkerModel.from_pretrained(model_path, labels=labels)
    accepted = ["span", "bert"]
    if any([a in model_path for a in accepted]):
        print(f"Using {model_path} as tokenizer")
        tokenizer = SpanMarkerTokenizer.from_pretrained(model_path, config=model.tokenizer.config)
    else:
        print("Using Roberta as tokenizer")
        tokenizer = SpanMarkerTokenizer.from_pretrained("roberta-base", config=model.tokenizer.config)
        model.set_tokenizer(tokenizer)

    def compute_f1(eval_prediction):
        is_in_train=False
        inputs = eval_prediction.inputs
        gold_labels = eval_prediction.label_ids
        logits = eval_prediction.predictions[0]
        num_words = eval_prediction.predictions[2]
        has_document_context = len(eval_prediction.predictions) == 5
        if has_document_context:
            document_ids = eval_prediction.predictions[3]
            sentence_ids = eval_prediction.predictions[4]

        # Compute probabilities via softmax and extract 'winning' scores/labels
        probs = torch.tensor(logits, dtype=torch.float32).softmax(dim=-1)
        scores, pred_labels = probs.max(-1)

        # Collect all samples in one dict. We do this because some samples are spread between multiple inputs
        sample_list = []
        for sample_idx in range(inputs.shape[0]):
            tokens = inputs[sample_idx]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            token_hash = hash(text) if not has_document_context else (document_ids[sample_idx], sentence_ids[sample_idx])
            if (
                not sample_list
                or sample_list[-1]["hash"] != token_hash
                or len(sample_list[-1]["spans"]) == len(sample_list[-1]["gold_labels"])
            ):
                mask = gold_labels[sample_idx] != -100
                spans = list(tokenizer.get_all_valid_spans(num_words[sample_idx], tokenizer.config.entity_max_length))
                sample_list.append(
                    {
                        "text": text,
                        "gold_labels": gold_labels[sample_idx][mask].tolist(),
                        "pred_labels": pred_labels[sample_idx][mask].tolist(),
                        "scores": scores[sample_idx].tolist(),
                        "num_words": num_words[sample_idx],
                        "hash": token_hash,
                        "spans": spans,
                    }
                )
            else:
                mask = gold_labels[sample_idx] != -100
                sample_list[-1]["gold_labels"] += gold_labels[sample_idx][mask].tolist()
                sample_list[-1]["pred_labels"] += pred_labels[sample_idx][mask].tolist()
                sample_list[-1]["scores"] += scores[sample_idx].tolist()

        outside_id = tokenizer.config.outside_id
        id2label = tokenizer.config.id2label
        pp = []
        ll = []
        for sample in sample_list:
            scores = sample["scores"]
            num_words = sample["num_words"]
            spans = sample["spans"]
            gold_labels = sample["gold_labels"]
            pred_labels = sample["pred_labels"]
            assert len(gold_labels) == len(pred_labels) and len(spans) == len(pred_labels)

            # Construct IOB2 format for gold labels, useful for seqeval
            gold_labels_per_tokens = ["O"] * num_words
            for span, gold_label in zip(spans, gold_labels):
                if gold_label != outside_id:
                    gold_labels_per_tokens[span[0]] = "B-" + id2label[gold_label]
                    gold_labels_per_tokens[span[0] + 1 : span[1]] = ["I-" + id2label[gold_label]] * (span[1] - span[0] - 1)

            # Same for predictions, note that we place most likely spans first and we disallow overlapping spans for now.
            pred_labels_per_tokens = ["O"] * num_words
            for _, span, pred_label in sorted(zip(scores, spans, pred_labels), key=lambda tup: tup[0], reverse=True):
                if pred_label != outside_id and all(pred_labels_per_tokens[i] == "O" for i in range(span[0], span[1])):
                    pred_labels_per_tokens[span[0]] = "B-" + id2label[pred_label]
                    pred_labels_per_tokens[span[0] + 1 : span[1]] = ["I-" + id2label[pred_label]] * (span[1] - span[0] - 1)
            pp.append(pred_labels_per_tokens)
            ll.append(gold_labels_per_tokens)

        unique_labels = list(set([l.split("-")[-1] for l in entities]))
        evaluator = Evaluator(
                pp, ll, tags=unique_labels, loader="list"
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
    
    # See the 🤗 TrainingArguments documentation for details here
    args = TrainingArguments(
        output_dir="legal-span-marker",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=acc_step,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit = 1,
        load_best_model_at_end=True,
        logging_steps=200,
        metric_for_best_model="f1-strict",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        report_to="wandb"
    )

    # Our Trainer subclasses the 🤗 Trainer, and the usage is very similar
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_f1
    )

    # Training is really simple using our Trainer!
    trainer.train()

    # ... and so is evaluating!
    metrics = trainer.evaluate()
    print(metrics)

    

"""python 3.10
Example of usage:
python main_span.py \
    --ds_path
    --output_folder results/ \
    --batch 256 \
    --num_epochs 5 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06
"""