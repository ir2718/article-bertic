from src.metric_learning.metric_model import get_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.train_ml_baseline import SVMModel, ExplosionEmbeddings
from time import time
import numpy as np
import pandas as pd
import pickle
import spacy
import torch

multi_bert = get_model(
    loss="cosine_similarity",
    model_name="bert-base-multilingual-cased",
    embedding_size=None,
    device="cuda",
    pooling_type="mean",
    distributed=True
)
multi_bert.load_best_model()
# 
bertic_ner_ce = AutoModelForSequenceClassification.from_pretrained(
    "./models/classla/bcms-bertic-ner/checkpoint-240", local_files_only=True
).to("cuda:0")
bertic_ner_ce_tokenizer = AutoTokenizer.from_pretrained("classla/bcms-bertic-ner")

svm = pickle.load(open("./models/svm_spacy_hyperopt/model.pkl", "rb"))
embeddings_object = ExplosionEmbeddings()

num_repeats = 10
df = pd.read_csv("./data/similarity_dataset/similarity_dataset.csv")

num_examples = [10, 100, 1000]
for num in num_examples:
    text_of_interest = df.iloc[0]["body"]
    df_subset = df.iloc[np.random.choice(list(range(1, df.shape[0])), size=(num,))]["body2"].values.tolist()

    # multi_bert
    encoded_ex = multi_bert.encode(text_of_interest) 
    encoded_db = [multi_bert.encode(i) for i in df_subset]
    times_multi_bert = []
    for r in range(num_repeats):
        start = time()
        for encoded_other in encoded_db:
            _ = encoded_ex @ encoded_other.T
        end = time()
        times_multi_bert.append(end - start)
    mean_multi_bert, stdev_multi_bert = np.mean(times_multi_bert), np.std(times_multi_bert)
    print(f"Time for multilingual BERT on {num_repeats} runs for size {num}: {mean_multi_bert} +/- {stdev_multi_bert}")

    # bertic ce
    times_bertic_ner_ce = []
    for r in range(num_repeats):
        start = time()
        for i in df_subset:
            tokenized_example = bertic_ner_ce_tokenizer(text_of_interest, i, padding=True, truncation=True, return_tensors="pt")
            tokenized_example = {k:v.to("cuda:0") for k, v in tokenized_example.items()}
            _ = bertic_ner_ce(**tokenized_example)
        end = time()
        times_bertic_ner_ce.append(end - start)
    mean_bertic_ner_ce, stdev_bertic_ner_ce = np.mean(times_bertic_ner_ce), np.std(times_bertic_ner_ce)
    print(f"Time for BERTic NER CE on {num_repeats} runs for size {num}: {mean_bertic_ner_ce} +/- {stdev_bertic_ner_ce}")

    # svm
    times_svm = []
    df_repeated = pd.DataFrame.from_dict({
        "body": [text_of_interest for _ in range(num)],
        "body2": df_subset
    })
    embedding_means = embeddings_object(df_repeated)
    for r in range(num_repeats):
        start = time()
        for i in range(num):
            _ = svm(embedding_means[i, :].reshape(1, -1))
        end = time()
        times_svm.append(end - start)
    mean_svm, stdev_svm = np.mean(times_svm), np.std(times_svm)
    print(f"Time for SVM on {num_repeats} runs for size {num}: {mean_svm} +/- {stdev_svm}")
    print("\n"*3)