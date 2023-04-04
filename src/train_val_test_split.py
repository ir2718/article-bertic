import argparse
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# python -m src.train_val_test_split --save_path ./data/similarity_dataset/ --dataset_path ./data/similarity_dataset/similarity_dataset.csv --val_perc 0.2 --test_perc 0.1

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--val_perc", type=float, default=0.2, required=True)
    parser.add_argument("--test_perc", type=float, default=0.1, required=True)
    parser.add_argument("--seed", type=float, default=42)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    np.random.seed(args.seed)

    df = pd.read_csv(args.dataset_path)

    # x = test_perc / (1 - val_perc)
    #
    # x = 0.1 / (1 - 0.2) = 0.125 -> 0.125 * 0.8 = 0.1 

    df_train, df_val = train_test_split(
        df, stratify=df["choice"], test_size=args.val_perc
    )

    perc_in_remaining = args.test_perc / (1 - args.val_perc)
    df_train, df_test = train_test_split(
        df_train, stratify=df_train["choice"], test_size=perc_in_remaining
    )

    df_train.to_csv(os.path.join(args.save_path, "similarity_train_dataset.csv"), index=False)
    df_val.to_csv(os.path.join(args.save_path, "similarity_validation_dataset.csv"), index=False)
    df_test.to_csv(os.path.join(args.save_path, "similarity_test_dataset.csv"), index=False)

    print(f"     Train:  {df_train.shape}")
    print(f"Validation:  {df_val.shape}")
    print(f"      Test:  {df_test.shape}")