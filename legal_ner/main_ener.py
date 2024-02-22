import os
import torch
import re
import evaluate
import numpy as np

from argparse import ArgumentParser
from nervaluate import Evaluator
from peft import LoraConfig, TaskType, get_peft_model, AdaLoraConfig, IA3Config

from transformers import AutoModelForTokenClassification
from transformers import Trainer, DefaultDataCollator, TrainingArguments, DataCollatorForTokenClassification, DataCollatorWithPadding

from utils.dataset import LegalNERTokenDataset, load_legal_ner, ENER_Dataset

from span_marker import SpanMarkerModel, Trainer as SpanTrainer
from span_marker.tokenizer import SpanMarkerTokenizer

# SET SEED FOR REPRODUCIBILITY
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
############################################################
#                                                          #
#                           MAIN                           #
#                                                          #
############################################################ 
if __name__ == "__main__":
    seqeval = evaluate.load("seqeval")
    parser = ArgumentParser(description="Training of LUKE model")
    parser.add_argument(
        "--dataset",
        help="Choose dataset",
        default="legal_ner",
        required=False,
        choices=["legal_ner", "ener"],
        type=str,
    )
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
        default="bert-base",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--use_span",
        help="Use Span Model",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--push_to_hub",
        help="Push the model to the hub once the training is finished",
        default=False,
        required=False,
        action="store_true"
    )
    parser.add_argument(
        "--hub_token",
        help="Huggingface write token",
        default=None,
        required=False,
        type=str
    )
    parser.add_argument(
        "--hub_model_id",
        help="Huggingface full path",
        default=None,
        required=False,
        type=str
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

    parser.add_argument(
        "--lora_rank",
        help="Lora Rank",
        default=16,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--lora_alpha",
        help="Lora Alpha",
        default=8,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--lora_dropout",
        help="Lora Dropout",
        default=0.1,
        required=False,
        type=float,
    )

    parser.add_argument(
        "--peft_mode",
        help="Choice of PEFT algorithm",
        required=False,
        type=str,
        choices=["lora", "adalora", "ia3"],
        default=None
    )

    parser.add_argument(
        "--lora_bias",
        help="Lora bias",
        required=False,
        type=str,
        choices=["all", "none"],
        default="all"
    )

    parser.add_argument(
        "--target_modules",
        help="Which modules to target with Lora",
        required=False,
        type=str,
        choices=["default", "linear"],
        default="default"
    )

    parser.add_argument(
        "--wxe",
        help="How many words per entity to use with SpanMarker",
        required=False,
        type=int,
        default=6
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
    model_path = args.model_path
    use_span = args.use_span
    acc_step = args.acc_step
    scheduler = args.scheduler
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    peft_mode = args.peft_mode
    lora_dropout = args.lora_dropout
    bias = args.lora_bias
    dataset = args.dataset
    push_to_hub = args.push_to_hub
    hub_token = args.hub_token
    hub_model_id = args.hub_model_id
    target_modules = args.target_modules
    words_per_entity = args.wxe

    if use_span:
        print("Span Mode Activated")
    
    if dataset == "legal_ner":
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
        
    else:
        original_label_list = [
            "BUSINESS", 
            "LOCATION",
            "PERSON", 
            "GOVERNMENT", 
            "COURT", 
            "LEGACT", 
            "MISCELLANEOUS"
        ]
    
    labels_list = ["B-" + l for l in original_label_list]
    labels_list += ["I-" + l for l in original_label_list]
    span_labels = ["O"]+labels_list
    num_labels = len(labels_list) + 1

    def compute_metrics_ener(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [idx_to_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [idx_to_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
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

    def compute_score_span(eval_prediction):
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

        unique_labels = list(set([l.split("-")[-1] for l in labels_list]))
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
    


    print("MODEL: ", model_path)
    if not use_span:
        ## Define the train and test datasets
        use_roberta = False
        if "luke" in model_path or "roberta" in model_path:
            use_roberta = True

        if dataset == "ener":
            data_processor = ENER_Dataset(ds_train_path, ds_valid_path, labels_list=labels_list, tokenizer=model_path)
            tok_dataset = data_processor.get_ener_dataset()
            idx_to_labels = {v[1]: v[0] for v in data_processor.labels_to_idx.items()}
        else:
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
            idx_to_labels = {v[1]: v[0] for v in train_ds.labels_to_idx.items()}
            

        ## Define the model
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, 
            num_labels=num_labels, 
            ignore_mismatched_sizes=True
        )
        ## Map the labels
        
    else:
        
        model = SpanMarkerModel.from_pretrained(
            model_path, 
            labels=span_labels, 
            model_max_length=128,
            marker_max_length=64,
            entity_max_length=words_per_entity
        )

        accepted = ["span", "bert"]
        if any([a in model_path for a in accepted]) and ("luke" not in model_path):
            print(f"Using {model_path} as tokenizer")
            tokenizer = SpanMarkerTokenizer.from_pretrained(model_path, config=model.config)
        else:
            print("Using Roberta as tokenizer")
            tokenizer = SpanMarkerTokenizer.from_pretrained("roberta-base", config=model.config)

        print(tokenizer.tokenizer)
        #model.set_tokenizer(tokenizer)
        
        if dataset =="legal_ner":
            span_dataset = load_legal_ner(ds_train_path, ds_valid_path)
        else:
            data_processor = ENER_Dataset(ds_train_path, ds_valid_path, tokenizer = model.tokenizer.tokenizer, labels_list=labels_list)
            span_dataset = data_processor.get_ener_dataset_e()
            
    print(model)
    
    if peft_mode is not None:
        if "luke" in model_path and target_modules=="default":
            target_modules = ['query', 'e2w_query', 'e2e_query', 'value', 'w2e_query']
            print(f"Found target modules: \n{target_modules}")
        elif target_modules=="linear":
            print("Applying Lora to all linear layers")
            model_modules = str(model.modules)
            pattern = r'\((\w+)\): Linear'
            linear_layer_names = re.findall(pattern, model_modules)

            names = []
            # Print the names of the Linear layers
            for name in linear_layer_names:
                names.append(name)
            target_modules = list(set(names))
            print(target_modules)
        else:
            target_modules = None
        if peft_mode == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS, inference_mode=False, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias, target_modules=target_modules
            )
        elif peft_mode == "adalora":
            peft_config = AdaLoraConfig(
                target_r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type=TaskType.TOKEN_CLS,
                inference_mode=False,
                target_modules=target_modules
            )
        else:
            peft_config = IA3Config(
                task_type=TaskType.TOKEN_CLS,
                target_modules=target_modules+["output.dense"],
                feedforward_modules=["output.dense"]
            )

        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
        print(model)


    ## Output folder
    new_output_folder = os.path.join(output_folder, 'all')
    new_output_folder = os.path.join(new_output_folder, model_path)
    if not os.path.exists(new_output_folder):
        os.makedirs(new_output_folder)

    if not use_span:
        ## Training Arguments
        training_args = TrainingArguments(
            output_dir=new_output_folder,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=acc_step,
            gradient_checkpointing=peft_mode is None,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            fp16=False,
            fp16_full_eval=False,
            metric_for_best_model="f1-strict" if dataset!="ener" else "f1",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to="wandb",
            logging_steps=50,  # how often to log to W&B
            lr_scheduler_type=scheduler,
            push_to_hub=push_to_hub,
            hub_token=hub_token,
            hub_model_id=hub_model_id
        )

        ## Collator
        data_collator = DefaultDataCollator() if dataset != "ener" else DataCollatorForTokenClassification(tokenizer=data_processor.tokenizer)


        ## Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds if dataset!="ener" else tok_dataset["train"],
            eval_dataset=val_ds if dataset!="ener" else tok_dataset["test"],
            compute_metrics=compute_metrics if dataset!="ener" else compute_metrics_ener,
            data_collator=data_collator,
        )

    else:
        training_args = TrainingArguments(
            output_dir=new_output_folder,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=acc_step,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            fp16=False,
            fp16_full_eval=False,
            metric_for_best_model="f1-strict",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to="wandb",
            logging_steps=50,  # how often to log to W&B
            lr_scheduler_type=scheduler,
            push_to_hub=push_to_hub,
            hub_token=hub_token,
            hub_model_id=hub_model_id
        )
        # Our Trainer subclasses the 🤗 Trainer, and the usage is very similar
        trainer = SpanTrainer(
            model=model,
            args=training_args,
            train_dataset=span_dataset["train"],
            eval_dataset=span_dataset["dev"] if dataset=="legal_ner" else span_dataset["test"],
            compute_metrics=compute_score_span,
        )


    ## Train the model and save it
    trainer.train()
    trainer.save_model(output_folder)
    if push_to_hub:
        trainer.push_to_hub()
        if use_span:
            print("Pushing tok")
            trainer.tokenizer.push_to_hub(repo_id=hub_model_id, token=hub_token)
            trainer.tokenizer.tokenizer.push_to_hub(repo_id=hub_model_id, token=hub_token)
            trainer.tokenizer.save_pretrained("tok.json")
            trainer.tokenizer.tokenizer.save_pretrained("tok_tok.json")

        else:
            if dataset!="ener":
                train_ds.tokenizer.push_to_hub(repo_id=hub_model_id, token=hub_token)
            else:
                data_processor.tokenizer.push_to_hub(repo_id=hub_model_id, token=hub_token)



"""python 3.10
Example of usage (baseline):
python main.py \
    --ds_train_path data/NER_TRAIN/NER_TRAIN_ALL.json \
    --ds_valid_path data/NER_DEV/NER_DEV_ALL.json \
    --output_folder results/ \
    --batch 256 \
    --num_epochs 5 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --model_path studio-ousia/luke-base

Example of usage (baseline):
python main.py (Ours) \
    --ds_train_path data/NER_TRAIN/NER_TRAIN_ALL.json \
    --ds_valid_path data/NER_DEV/NER_DEV_ALL.json \
    --output_folder results/ \
    --batch 8 \
    --num_epochs 5 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --model_path studio-ousia/luke-base \
    --span
"""