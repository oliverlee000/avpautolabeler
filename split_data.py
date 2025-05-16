import pandas as pd
import numpy as np
import argparse
import os

def split_dataset(input_csv, seed=42):
    # Set seed
    np.random.seed(seed)

    # Read CSV
    df = pd.read_csv(input_csv)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Compute sizes
    n = len(df)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)

    # Split
    df_train = df[:n_train]
    df_dev = df[n_train:n_train + n_dev]
    df_test = df[n_train + n_dev:]

    # Ensure output folder exists
    os.makedirs("data", exist_ok=True)

    # Save
    df_train.to_csv("data/train.csv", index=False)
    df_dev.to_csv("data/dev.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split_dataset(args.input_csv, seed=args.seed)
