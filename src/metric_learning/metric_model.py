from torch.nn.functional import (
    cross_entropy, 
    cosine_similarity, 
    mse_loss,
    binary_cross_entropy_with_logits,
    softmax,
    normalize
)
from transformers import AutoModel, AutoTokenizer, AutoConfig
from src.metric_learning.enums import Loss, Pooling
from scipy.stats import pearsonr, spearmanr
from abc import ABC, abstractmethod
from typing import Tuple
from tqdm import tqdm
from src.metric_learning.train_utils import get_optimizer, get_scheduler
import torch.nn as nn
import numpy as np
import torch
import json
import os

def get_model(loss, **model_kwargs):
    d = {
        Loss.CROSS_ENTROPY_LOSS: CrossEntropyArticleEmbeddingModel,
        Loss.COSINE_SIMILARITY_LOSS: CosineSimilarityArticleEmbeddingModel,
    }

    model = d[loss](loss_name=loss, **model_kwargs)

    return model

def get_pooling_layer(pooling_type):
    d = {
        Pooling.MEAN_POOLING: MeanPooling
    }
    return d[pooling_type]()

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        emb_sum = torch.sum(last_hidden * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9) # denominator
        emb_mean = emb_sum / sum_mask
        return emb_mean
    
class BaseArticleEmbeddingModel(ABC, nn.Module):

    MAX_GRAD_NORM = 1.0
    METRIC = "spearman"

    def __init__(self, loss_name, model_name, pooling_type, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.model_name = model_name
        self.loss_name = loss_name

        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, config=self.config, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        if self.embedding_size is None:
            self.embedding_layer = nn.Identity()
        else:
            self.embedding_layer = nn.Sequential(
                nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size),
                nn.GELU(),
                nn.Linear(in_features=self.config.hidden_size, out_features=self.embedding_size),
            )

        self.pooling_layer = get_pooling_layer(pooling_type)
        
        self.device = device
        self.to(device)

    def set_optimizer(self, optimizer_type, lr, weight_decay):
        self.optimizer = get_optimizer(
            model=self,
            optimizer_type=optimizer_type, 
            lr=lr, 
            weight_decay=weight_decay
        )

    def set_scheduler(self, scheduler_type, num_training_steps, warmup_ratio):
        self.scheduler = get_scheduler(
            scheduler_type=scheduler_type, 
            optimizer=self.optimizer,
            num_training_steps=num_training_steps, 
            warmup_ratio=warmup_ratio
        )

    def forward_once(self, inputs):
        out = self.model(**inputs).last_hidden_state
        out_mean = self.pooling_layer(out, inputs["attention_mask"])
        out_emb = self.embedding_layer(out_mean)
        out_emb_norm = normalize(out_emb, p=2.0, dim=1)
        return out_emb_norm

    def forward(self, input1, input2):
        # get two texts features
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)

        # combine them in subclass
        out = self.combine_features(out1, out2)

        # return unnormalized output
        return out
    
    def init_metric_dicts(self):
        self.test_metric_dict = {"spearman": [], "pearson": []}
        self.val_metric_dict = {"spearman": [], "pearson": []}
        self.train_metric_dict = {"spearman": [], "pearson": []}
        
    def compute_metrics(self, outputs, targets):
        outputs = self.postprocess(outputs).detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        
        return {
            "spearman": spearmanr(outputs, targets)[0],
            "pearson": pearsonr(outputs, targets)[0]
        }
        
    def inputs_to_device(self, batch):
        return {k:v.to(self.device) for k, v in batch.items()}
        
    def train_metric_model(self, num_epochs, train_dataloader, validation_dataloader, gradient_accumulation_steps):
        self.train()
        self.init_metric_dicts()

        step = 0
        for e in range(num_epochs):
            for i, batch in tqdm(enumerate(train_dataloader)):
                text1, text2, target = batch

                text1_inputs = self.tokenizer(text1, padding=True, truncation=True, return_tensors="pt")
                text2_inputs = self.tokenizer(text2, padding=True, truncation=True, return_tensors="pt")
                
                text1_inputs = self.inputs_to_device(text1_inputs)
                text2_inputs = self.inputs_to_device(text2_inputs)
                target = target.to(self.device).float()
                
                out = self.forward(text1_inputs, text2_inputs)
                
                loss = self.loss(out, target) / gradient_accumulation_steps
                loss.backward()

                step += 1
                if step % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.parameters(), BaseArticleEmbeddingModel.MAX_GRAD_NORM)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.zero_grad()

            self.validate_model(train_dataloader, validation_dataloader)
            self.log_metrics()
            
            if self.val_metric_dict[BaseArticleEmbeddingModel.METRIC][-1] == max(self.val_metric_dict[BaseArticleEmbeddingModel.METRIC]):
                self.save_model(e)
        
        self.load_best_model()
        return self
    
    
    def log_metrics(self):
        print()
        for k in self.train_metric_dict.keys():
            print(f"Train {k} coefficient - {np.round(self.train_metric_dict[k][-1], decimals=3)}")
            
        for k in self.val_metric_dict.keys():
            print(f"Validation {k} coefficient - {np.round(self.val_metric_dict[k][-1], decimals=3)}")
        print()
        
    
    @torch.no_grad()
    def validate_dataloader(self, dataloader):
        preds, targets = [], []
        for i, batch in tqdm(enumerate(dataloader)):
            text1, text2, target = batch

            text1_inputs = self.tokenizer(text1, padding=True, truncation=True, return_tensors="pt")
            text2_inputs = self.tokenizer(text2, padding=True, truncation=True, return_tensors="pt")
            
            text1_inputs = self.inputs_to_device(text1_inputs)
            text2_inputs = self.inputs_to_device(text2_inputs)
            target = target.to(self.device)
            
            out = self.forward(text1_inputs, text2_inputs)
            
            preds.extend(out)
            targets.extend(target)
            
        return torch.stack(preds), torch.stack(targets)
        
    def validate_model(self, train_dataloader, validation_dataloader):
        self.eval()

        print("Calculating metrics on train . . .")
        train_preds, train_targets = self.validate_dataloader(train_dataloader)
        train_metrics = self.compute_metrics(train_preds, train_targets)

        print("Calculating metrics on validation . . .")
        val_preds, val_targets = self.validate_dataloader(validation_dataloader)        
        val_metrics = self.compute_metrics(val_preds, val_targets)
        
        for k in train_metrics.keys():
            self.train_metric_dict[k].append(train_metrics[k])
            self.val_metric_dict[k].append(val_metrics[k])
        
        self.train()

    def test_model(self, test_dataloader):
        self.eval()
        
        print("Calculating metrics on test . . .")
        test_preds, test_targets = self.validate_dataloader(test_dataloader)
        test_metrics = self.compute_metrics(test_preds, test_targets)
        
        for k in test_metrics.keys():
            self.test_metric_dict[k].append(test_metrics[k])

    def postprocess(self, out):
        # used only in cross entropy to add softmax
        return out

    def load_best_model(self):
        save_path = f"./models/article_bertic/{self.model_name}/{self.loss_name}"
        model_save_path = [i for i in os.listdir(save_path) if i.endswith(".pt")][0]
        print(f"Found best model at path: {model_save_path}")
        self.load_state_dict(torch.load(os.path.join(save_path, model_save_path)))

    def save_model(self, epoch):
        save_path = f"./models/article_bertic/{self.model_name}/{self.loss_name}"
        model_save_path = f"{save_path}/best_model.pt"
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), model_save_path)
        print()
        print(f"Saved the model to {model_save_path}")
        print()

    def save_all_metrics(self):
        save_path = f"./models/article_bertic/{self.model_name}/{self.loss_name}/metrics.json"
        
        all_metrics = {}
        for k in self.train_metric_dict.keys():
            all_metrics[f"train_{k}"] = self.train_metric_dict[k]
            all_metrics[f"validation_{k}"] = self.val_metric_dict[k]
            all_metrics[f"test_{k}"] = self.test_metric_dict[k]

        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp) 
    
        print()
        print(f"Saved the metrics to {save_path}")
        print()

    @abstractmethod
    def combine_features(out1, out2) -> torch.Tensor:
        pass

    @abstractmethod
    def loss(self, output, target) -> torch.Tensor:
        pass


class CrossEntropyArticleEmbeddingModel(BaseArticleEmbeddingModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = nn.Sequential(
            nn.Linear(self.config.hidden_size * 3, 1), # u, v, |u - v|
        ).to(self.device)
        
    def postprocess(self, out):
        num_labels = out.shape[-1]
        
        if num_labels == 1:
            return torch.sigmoid(out.view(-1))
        
        return softmax(out)

    def combine_features(self, out1, out2):
        features = torch.cat((out1, out2, torch.abs(out1 - out2)), dim=1)
        return self.cls(features)

    def loss(self, output, target):
        num_labels = output.shape[-1]
        
        if num_labels == 1:
            return binary_cross_entropy_with_logits(output.view(-1), target.view(-1))
        
        return cross_entropy(output, target)
    

class CosineSimilarityArticleEmbeddingModel(BaseArticleEmbeddingModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def combine_features(self, out1, out2):
        return cosine_similarity(out1, out2)

    def loss(self, output, target):
        return mse_loss(output, target)


if __name__ == "__main__":
    model = CrossEntropyArticleEmbeddingModel("classla/bcms-bertic", 512)
    pass