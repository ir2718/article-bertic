from src.metric_learning.similarity_dataset import get_dataset
from src.metric_learning.metric_model import get_model
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import set_seed
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import argparse

def parse_visualization():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="classla/bcms-bertic")
    parser.add_argument("--loss_function", type=str, default="cosine_similarity")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--embedding_size", default=None)
    parser.add_argument("--num_clusters", type=int, default=8)
    parser.add_argument("--pooling_type", type=str, default="mean")
    parser.add_argument("--dataset_root", type=str, default="./data/similarity_dataset")
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return args

args = parse_visualization()
set_seed(args.seed)

model = get_model(
    loss=args.loss_function,
    model_name=args.model_name,
    embedding_size=args.embedding_size,
    device=args.device,
    pooling_type=args.pooling_type,
    distributed=args.distributed
)
model.load_best_model()

test_dataset = get_dataset(loss=args.loss_function, root=args.dataset_root, split="test")
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size)

model.eval()

embeddings = []
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_dataloader)):
        text1, text2, _ = batch
        
        text1_inputs = model.tokenizer(text1, padding=True, truncation=True, return_tensors="pt")
        text2_inputs = model.tokenizer(text2, padding=True, truncation=True, return_tensors="pt")
        
        text1_inputs = model.inputs_to_device(text1_inputs)
        text2_inputs = model.inputs_to_device(text2_inputs)
        
        out1 = model.forward_once(text1_inputs).cpu().detach().numpy()
        out2 = model.forward_once(text2_inputs).cpu().detach().numpy()
                    
        embeddings.extend(out1.tolist())
        embeddings.extend(out2.tolist())
    
embeddings = np.array(embeddings)


emb_clusters = KMeans(n_clusters=args.num_clusters, init="k-means++").fit(embeddings).labels_
tsne_embeddings = TSNE(perplexity=args.perplexity, n_iter=args.n_iter, metric="cosine").fit_transform(embeddings)

if args.num_clusters < 0:
    inertia = []
    for i in range(1, 40):
        kmeans = KMeans(n_clusters=i, init="k-means++")
        x_cluster = kmeans.fit_transform(tsne_embeddings)
        inertia.append(kmeans.inertia_)
    plt.plot(range(1, len(inertia) + 1), inertia)
    plt.show()
    exit(0)


plt.figure(figsize=(6, 6))
plt.axis("equal")
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1],c=emb_clusters)
plt.savefig(
    os.path.join("./figures", f"{args.model_name.replace('/', '-')}_{args.loss_function}.png")
)
plt.show()
