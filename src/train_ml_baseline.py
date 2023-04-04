import fasttext
import pandas as pd
import numpy as np
import json
import argparse
import spacy
import torch
import os
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from ast import literal_eval

class KNNModel:

    def __init__(self, **kwargs):
        self.model = KNeighborsRegressor(**kwargs)
    
    @staticmethod
    def scale_labels(y):
        return y

    def fit(self, x, y):
        self.model.fit(x, y)

    def __call__(self, x):
        return self.model.predict(x)

    @staticmethod
    def hyperparam_dict():
        return {
            "n_neighbors": list(range(3, 30)),
            "weights": ["uniform", "distance"],
            "metric": ["l1", "l2", "cosine"]
        }
    

class SVMModel:

    def __init__(self, **kwargs):
        self.model = SVR(**kwargs)
    
    @staticmethod
    def scale_labels(y):
        return y

    def fit(self, x, y):
        self.model.fit(x, y)

    def __call__(self, x):
        return self.model.predict(x)

    @staticmethod
    def hyperparam_dict():
        return {
            "kernel":["linear", "poly", "rbf", "sigmoid"],
            "degree": list(range(2,8)),
            "C": np.array([[float(j*i) for i in range(1, 6)] for j in [1e-4, 1e-3, 1e-2, 1e-1, 1]]).flatten()
        }

class LRModel:

    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)

    @staticmethod
    def scale_labels(y):
        y_tmp = np.clip(y/5., 1e-8, 1 - 1e-8)
        y_tmp_inv = np.log(y_tmp / (1 - y_tmp))
        return y_tmp_inv

    def fit(self, x, y):
        # important to scale the labels to [0, 1]
        y_tmp = np.clip(y/5., 1e-8, 1 - 1e-8)
        y_tmp_inv = np.log(y_tmp / (1 - y_tmp))

        self.model.fit(x, y_tmp_inv)

    def __call__(self, x):
        out_x = self.model.predict(x)

        # to avoid numerical overflow
        lr_out = np.where(
            out_x >= 0,  
            1/(1 + np.exp(-out_x)),  
            np.exp(out_x) / (1 + np.exp(out_x))
        )
        
        # convert back to [0, 5] when predicting
        return lr_out * 5

    @staticmethod
    def hyperparam_dict():
        return {}

class LRL2Model:

    def __init__(self, **kwargs):
        self.model = Ridge(**kwargs)

    @staticmethod
    def scale_labels(y):
        y_tmp = np.clip(y/5., 1e-8, 1 - 1e-8)
        y_tmp_inv = np.log(y_tmp / (1 - y_tmp))
        return y_tmp_inv

    def fit(self, x, y):
        # important to scale the labels to [0, 1]
        y_tmp = np.clip(y/5., 1e-8, 1 - 1e-8)
        y_tmp_inv = np.log(y_tmp / (1 - y_tmp))

        self.model.fit(x, y_tmp_inv)

    def __call__(self, x):
        out_x = self.model.predict(x)

        # to avoid numerical overflow
        lr_out = np.where(
            out_x >= 0,  
            1/(1 + np.exp(-out_x)),  
            np.exp(out_x) / (1 + np.exp(out_x))
        )
        
        # convert back to [0, 5] when predicting
        return lr_out * 5

    @staticmethod
    def hyperparam_dict():
        return {
            "alpha": np.array([[float(j*i) for i in range(1, 6)] for j in [1e-4, 1e-3, 1e-2, 1e-1, 1]]).flatten(),
            "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "lsqr", "sag"] 
        }

class FastTextEmbeddings:

    def __init__(self):
        self.model = fasttext.load_model("./models/embeddings/cc.hr.300.bin")

    def __call__(self, text):
        text1 = text["body"].values
        text2 = text["body2"].values

        text1_embeddings, text2_embeddings = [], []
        for t1, t2 in tqdm(zip(text1, text2)):
            text1_embeddings.append(np.mean([self.model.get_word_vector(i) for i in t1.split()], axis=0))
            text2_embeddings.append(np.mean([self.model.get_word_vector(i) for i in t2.split()], axis=0))
        text1_embeddings, text2_embeddings = np.array(text1_embeddings), np.array(text2_embeddings)  

        return np.concatenate((text1_embeddings, text2_embeddings), axis=1)
        
class ExplosionEmbeddings:

    def __init__(self):
        self.model = spacy.load("hr_core_news_lg")

    def __call__(self, text):
        text1 = text["body"].values
        text2 = text["body2"].values

        text1_embeddings, text2_embeddings = [], []
        for t1, t2 in tqdm(zip(text1, text2)):
            text1_embeddings.append(self.model(t1).vector)
            text2_embeddings.append(self.model(t2).vector)
        text1_embeddings, text2_embeddings = np.array(text1_embeddings), np.array(text2_embeddings)  

        return np.concatenate((text1_embeddings, text2_embeddings), axis=1)

MODEL_DICT = {
    "knn": KNNModel,
    "svm": SVMModel,
    "lr": LRModel,
    "lrl2": LRL2Model,
}

EMBEDDING_DICT = {
    "fasttext": FastTextEmbeddings(),
    "spacy": ExplosionEmbeddings()
}

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(MODEL_DICT.keys()) + ["all"], required=True)
    parser.add_argument("--embeddings", type=str, choices=list(EMBEDDING_DICT.keys()))
    parser.add_argument("--save_path", type=str, default="./models")
    parser.add_argument("--hyperopt", type=literal_eval, default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

def compute_spearman(x, y):
    return spearmanr(x, y)[0]

def compute_metrics(x, y):
    return {
        "spearman": spearmanr(x, y)[0],
        "pearson": pearsonr(x, y)[0]
    }

def read_datasets():
    df_train = pd.read_csv("./data/similarity_dataset/similarity_train_dataset.csv")
    df_validation = pd.read_csv("./data/similarity_dataset/similarity_validation_dataset.csv")
    df_test = pd.read_csv("./data/similarity_dataset/similarity_test_dataset.csv")

    y_train = df_train["choice"]
    X_train = df_train.drop(columns=["choice"])

    y_validation = df_validation["choice"]
    X_validation = df_validation.drop(columns=["choice"])

    y_test = df_test["choice"]
    X_test = df_test.drop(columns=["choice"])

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def save_json(path, d):
    with open(path, 'w') as f:
        json.dump(d, f)
    f.close()

def train_model(model, embeddings, X_train, y_train, X_validation, y_validation, X_test, y_test, model_name, embedding_name, hyperopt):
    if not hyperopt:
        model = model()
    else:
        y_train_scaled, y_validation_scaled = model.scale_labels(y_train.values), model.scale_labels(y_validation.values)
        hp_dict = find_best_hyperparams(
            model().model, model.hyperparam_dict(), X_train, y_train_scaled, X_validation, y_validation_scaled
        )
        model = model(**hp_dict)
        
    model.fit(X_train, y_train)

    train_preds, validation_preds, test_preds = model(X_train), model(X_validation), model(X_test)
    
    train_metrics = compute_metrics(train_preds, y_train.values)

    validation_metrics = compute_metrics(validation_preds, y_validation.values)
    test_metrics = compute_metrics(test_preds, y_test.values)

    path_metrics = os.path.join(args.save_path, f"{model_name}_{embedding_name}" if not hyperopt else f"{model_name}_{embedding_name}_hyperopt") 
        
    os.makedirs(path_metrics, exist_ok=True)

    all_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
    all_metrics.update({f"validation_{k}": v for k, v in validation_metrics.items()})
    all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

    save_json(os.path.join(path_metrics, "metrics.json"), all_metrics)
    if hyperopt:
        save_json(os.path.join(path_metrics, "hyperparams.json"), hp_dict)

    print(f"Saved results to {path_metrics}")
    print()

def find_best_hyperparams(model, parameters, X_train, y_train, X_validation, y_validation):
    X_whole = np.concatenate((X_train, X_validation), axis=0)
    y_whole = np.concatenate((y_train, y_validation), axis=0)

    # create mask of -1 where y_train else 0
    mask = np.zeros_like(y_whole)
    mask[:y_train.shape[0]] = -1

    # because of previously splitting the data
    cv = PredefinedSplit(mask)

    search = GridSearchCV(
        model, parameters, cv=cv, n_jobs=-1, 
        scoring=make_scorer(compute_spearman, greater_is_better=True)
    )
    search.fit(X_whole, y_whole)

    return search.best_params_

if __name__ == "__main__":
    args = parse()

    np.random.seed(args.seed)

    if args.model == "all":
        for e in EMBEDDING_DICT.keys():
            embeddings = EMBEDDING_DICT[e]
            X_train, y_train, X_validation, y_validation, X_test, y_test = read_datasets()

            X_train = embeddings(X_train[["body", "body2"]])
            X_validation = embeddings(X_validation[["body", "body2"]])
            X_test = embeddings(X_test[["body", "body2"]])

            for m in MODEL_DICT.keys():
                model = MODEL_DICT[m]

                print(f"Training {m} model using {e} embeddings . . .")
                train_model(
                    model, embeddings, X_train, y_train, X_validation, y_validation, X_test, y_test, m, e, args.hyperopt
                )
  
    else:
        model, embeddings = MODEL_DICT[args.model], EMBEDDING_DICT[args.embeddings]
        X_train, y_train, X_validation, y_validation, X_test, y_test = read_datasets()

        X_train = embeddings(X_train[["body", "body2"]])
        X_validation = embeddings(X_validation[["body", "body2"]])
        X_test = embeddings(X_test[["body", "body2"]])

        print(f"Training {args.model} model using {args.embeddings} embeddings . . .")
        train_model(
            model, embeddings, X_train, y_train, X_validation, y_validation, X_test, y_test, args.model, args.embeddings, args.hyperopt
        )

