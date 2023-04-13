from datasets import DatasetDict, Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoConfig, 
    AutoTokenizer, 
    set_seed, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from scipy.stats import spearmanr, pearsonr
from ast import literal_eval
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import datasets
import torch
import json
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="classla/bcms-bertic")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=int, default=1e-2)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--optimize_metric", type=str, default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "bce"])
    
    parser.add_argument("--dataset_path", type=str, default="./data/similarity_dataset")
    parser.add_argument("--save_path", type=str, default="./models")
    parser.add_argument("--fp16", type=literal_eval, default=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def read_datasets_as_hf(path):
    df_train = pd.read_csv(os.path.join(path, "similarity_train_dataset.csv"))
    df_train['choice'] = df_train['choice'].astype(np.float32)

    df_validation = pd.read_csv(os.path.join(path, "similarity_validation_dataset.csv"))
    df_validation['choice'] = df_validation['choice'].astype(np.float32)

    df_test = pd.read_csv(os.path.join(path, "similarity_test_dataset.csv"))
    df_test['choice'] = df_test['choice'].astype(np.float32)

    train_hf = Dataset.from_pandas(df_train)
    validation_hf = Dataset.from_pandas(df_validation)
    test_hf = Dataset.from_pandas(df_test)

    ds = DatasetDict()
    ds['train'] = train_hf.rename_column("choice", "labels")
    ds['validation'] = validation_hf.rename_column("choice", "labels")
    ds['test'] = test_hf.rename_column("choice", "labels")

    return ds

def compute_metrics(eval_pred):
    x, y = eval_pred[0].reshape(-1), eval_pred[1].reshape(-1)
    return {
        "spearman": spearmanr(x, y)[0],
        "pearson": pearsonr(x, y)[0]
    }

def preprocess_function(examples):
    return tokenizer(
        examples["body"], examples["body2"], 
        padding=True, truncation=True, truncation_strategy="longest_first", return_tensors="pt"
    )

def compute_dict_helper(preds, labels, problem_type, prefix):
    d = compute_metrics(
        (preds if problem_type == "mse" else torch.sigmoid(torch.tensor(preds).float()), labels)
    )

    d_prepend = {}
    for k, v in d.items():
        d_prepend[f"{prefix}_{k}"] = v
    
    return d_prepend

def evaluate_on_all_datasets(trainer, tokenized_datasets, output_dir, problem_type):
    train_preds = trainer.predict(tokenized_datasets["train"])
    validation_preds = trainer.predict(tokenized_datasets["validation"])
    test_preds = trainer.predict(tokenized_datasets["test"])

    all_metrics = {
        **compute_dict_helper(train_preds.predictions, train_preds.label_ids, problem_type, "train"),
        **compute_dict_helper(validation_preds.predictions, validation_preds.label_ids, problem_type, "validation"), 
        **compute_dict_helper(test_preds.predictions, test_preds.label_ids, problem_type, "test")
    }
    metric_file = os.path.join(output_dir, "metrics.json")

    with open(metric_file, 'w') as f:
        json.dump(all_metrics, f)
    f.close()
    print(f"Saved metrics to {metric_file}")

def scale_labels_for_bce(x):
    x["labels"] = [x["labels"] / 5.]
    return x

args = parse()
set_seed(args.seed)

if args.loss == "bce":
    num_labels = 1
    problem_type = "multi_label_classification"
elif args.loss == "mse":
    num_labels = 1
    problem_type = "regression"

config = AutoConfig.from_pretrained(
    args.model
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model, use_fast=False, 
)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=num_labels, problem_type=problem_type
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

ds = read_datasets_as_hf(args.dataset_path)
tokenized_ds = ds.map(preprocess_function, batched=True)

if args.loss == "bce":
    tokenized_ds["train"] = tokenized_ds["train"].map(scale_labels_for_bce) # scaling to [0, 1]
    tokenized_ds["validation"] = tokenized_ds["validation"].map(scale_labels_for_bce)
    tokenized_ds["test"] = tokenized_ds["test"].map(scale_labels_for_bce)


output_dir = os.path.join(args.save_path, args.model, args.loss)
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=args.lr,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.val_batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=args.weight_decay,
    optim=args.optim,
    metric_for_best_model=f"eval_{args.optimize_metric}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=args.fp16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


print("Training model . . .")
trainer.train()

print("Evaluating model . . .")
evaluate_on_all_datasets(trainer, tokenized_ds, output_dir, problem_type)

