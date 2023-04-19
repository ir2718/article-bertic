import pandas as pd
import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from src.metric_learning.enums import Loss

def get_dataset(loss, root, split):
    d = {
        Loss.CROSS_ENTROPY_LOSS: ZeroOneSimilarityDataset,
        Loss.COSINE_SIMILARITY_LOSS: ZeroOneSimilarityDataset,
    }
    
    dataset = d[loss](root=root, split=split)
    
    return dataset

class BaseSimilarityDataset(ABC, Dataset):
    
    LABEL_COL = "choice"
    FIRST_ARTICLE_COL = "body"
    SECOND_ARTICLE_COL = "body2"

    def __init__(self, root, split):
        super().__init__()
        self.df_split = pd.read_csv(os.path.join(root, f"similarity_{split}_dataset.csv"))
        self.split = split
        
        self.scale_labels()

    def __len__(self):
        return self.df_split.shape[0]
    
    def __getitem__(self, idx):
        row = self.df_split.iloc[idx]

        first_article = row[BaseSimilarityDataset.FIRST_ARTICLE_COL]
        second_article = row[BaseSimilarityDataset.SECOND_ARTICLE_COL]
    
        if BaseSimilarityDataset.LABEL_COL in row.index:
            label = row[BaseSimilarityDataset.LABEL_COL]

        return first_article, second_article, label
        
    @abstractmethod
    def scale_labels(self):
        pass
    
class ZeroOneSimilarityDataset(BaseSimilarityDataset):
    def scale_labels(self):
        if self.split == "train":
            self.df_split[ZeroOneSimilarityDataset.LABEL_COL] = self.df_split[ZeroOneSimilarityDataset.LABEL_COL] / 5.