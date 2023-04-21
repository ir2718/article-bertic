from src.metric_learning.parser import parse_visualization
from src.metric_learning.similarity_dataset import get_dataset
from torch.utils.data import DataLoader
from src.metric_learning.metric_model import load_model
from transformers import set_seed
from tqdm import tqdm
import numpy as np
import torch

args = parse_visualization()
set_seed(args.seed)

model = load_model(args.model_path)
test_dataset = get_dataset(loss=args.loss_function, root=args.dataset_root, split="test")
test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size)

model.eval()

embeddings = []
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

    with torch.no_grad():
        model.eval()
        test_rep = model.get_features(X.unsqueeze(1))
        test_rep2d = torch.pca_lowrank(test_rep, 2)[0]
        plt.scatter(test_rep2d[:, 0], test_rep2d[:, 1], color=colormap[Y[:]] / 255., s=5)
        plt.show()