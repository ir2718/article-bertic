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
import datasets
import argparse
import os
import pandas as pd
import numpy as np
import torch


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="classla/bcms-bertic")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=int, default=1e-2)
    
    parser.add_argument("--dataset_path", type=str, default="./data/similarity_dataset")
    parser.add_argument("--save_path", type=str, default="./models")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def read_datasets_as_hf(path):
    df_train = pd.read_csv(os.path.join(path, "similarity_train_dataset.csv"))
    df_validation = pd.read_csv(os.path.join(path, "similarity_validation_dataset.csv"))
    df_test = pd.read_csv(os.path.join(path, "similarity_test_dataset.csv"))

    train_hf = Dataset.from_pandas(df_train)
    validation_hf = Dataset.from_pandas(df_validation)
    test_hf = Dataset.from_pandas(df_test)

    ds = DatasetDict()
    ds['train'] = train_hf.rename_column("choice", "labels")
    ds['validation'] = validation_hf.rename_column("choice", "labels")
    ds['test'] = test_hf.rename_column("choice", "labels")

    return ds

def compute_metrics(x, y):
    return {
        "spearman": spearmanr(x, y)[0],
        "pearson": pearsonr(x, y)[0]
    }

def preprocess_function(examples):
    tok1 = tokenizer(examples["body"], padding=True, truncation=True, return_tensors="pt")
    tok2 = tokenizer(examples["body2"], padding=True, truncation=True, return_tensors="pt")

    return {
        "input_ids": torch.cat((
            tok1["input_ids"][:, :256,],  
            torch.full((tok1["input_ids"].shape[0], 1), tokenizer.sep_token_id), 
            tok2["input_ids"][:, 1:255], 
            torch.full((tok2["input_ids"].shape[0], 1), tokenizer.sep_token_id),
        ), dim=1).long(),
        "token_type_ids": torch.zeros_like(tok1["input_ids"]).float(),
        "attention_mask": torch.ones_like(tok1["input_ids"]).float()
    }

args = parse()
set_seed(args.seed)

config = AutoConfig.from_pretrained(
    args.model
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model, use_fast=False,
)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=1
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

ds = read_datasets_as_hf(args.dataset_path)
tokenized_ds = ds.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=os.path.join(args.save_path, args.model),
    learning_rate=args.lr,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.val_batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=args.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()