import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="classla/bcms-bertic")
    parser.add_argument("--embedding_size", default=None)
    parser.add_argument("--pooling_type", type=str, default="mean")
    parser.add_argument("--loss_function", type=str, default="cross_entropy") # cosine_similarity
    
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--dataset_root", type=str, default="./data/similarity_dataset")

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    return args

def parse_visualization():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="./data/similarity_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args
