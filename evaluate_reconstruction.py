"""Compare reconstructed source/target splits with real source/target partitions."""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def load_csv(path):
    """Load CSV and remove index column if present."""
    df = pd.read_csv(path)
    if df.columns[0] in {"Unnamed: 0", "index", ""}:
        df = df.iloc[:, 1:]
    return df


def match_data_to_indices(split_df, combined_df):
    """Match rows from split data to indices in combined dataset using exact matching after normalization."""
    split_cols = [c for c in split_df.columns if c not in ['target', 'y'] and str(c) not in ['Unnamed: 0', 'index', '']]
    combined_cols = [c for c in combined_df.columns if c not in ['class', 'y', 'target'] and str(c) not in ['Unnamed: 0', 'index', '']]
    
    combined_df_normalized = combined_df[combined_cols].copy()
    cont_cols = [col for col in combined_cols if combined_df[col].nunique() > 10]
    if cont_cols:
        combined_df_normalized[cont_cols] = StandardScaler().fit_transform(combined_df_normalized[cont_cols])
    
    combined_df_normalized['__idx__'] = combined_df.index
    merged = pd.merge(split_df[split_cols].round(10), combined_df_normalized[combined_cols + ['__idx__']].round(10), on=combined_cols, how='left')
    return merged['__idx__'].dropna().astype(int).values


def main():
    workspace = Path(__file__).parent
    data_dir = workspace / "data"
    split_dirs = sorted(workspace.glob("split_by_*_pareto_solutions"))
    
    if not split_dirs:
        print("No split_by_*_pareto_solutions directories found")
        return
    
    for pareto_dir in split_dirs:
        print(f"\nProcessing: {pareto_dir.name}")
        
        # Find solution files
        train_files = sorted(pareto_dir.glob("split_by_*_solution_*_train.csv"))
        if not train_files:
            continue
        
        sol_num = int(re.search(r"solution_(\d+)_train\.csv", train_files[-1].name).group(1))
        train_file = train_files[-1]
        test_file = train_file.parent / train_file.name.replace("_train.csv", "_test.csv")
        
        if not test_file.exists():
            continue
        
        # Find datasets
        datasets = {}
        for src_file in data_dir.glob("*_source.csv"):
            dataset_name = src_file.stem.replace("_source", "")
            tgt_file = data_dir / f"{dataset_name}_target.csv"
            if tgt_file.exists():
                datasets[dataset_name] = (src_file, tgt_file)
        
        for dataset_name, (source_file, target_file) in datasets.items():
            real_source = load_csv(source_file)
            real_target = load_csv(target_file)
            combined_df = pd.concat([real_source, real_target], ignore_index=True)
            
            source_size = len(real_source)
            true_labels = np.zeros(len(combined_df), dtype=int)
            true_labels[source_size:] = 1
            
            train_df = load_csv(train_file)
            test_df = load_csv(test_file)
            
            train_idx = match_data_to_indices(train_df, combined_df)
            test_idx = match_data_to_indices(test_df, combined_df)
            
            split_labels = np.full(len(combined_df), -1, dtype=int)
            split_labels[train_idx] = 0
            split_labels[test_idx] = 1
            
            mask = split_labels != -1
            if np.sum(mask) == 0:
                continue
            
            ari = adjusted_rand_score(true_labels[mask], split_labels[mask])
            
            source_in_train = np.sum((split_labels == 0) & (true_labels == 0))
            target_in_test = np.sum((split_labels == 1) & (true_labels == 1))
            train_size = np.sum(split_labels == 0)
            test_size = np.sum(split_labels == 1)
            
            print(f"  {dataset_name} - Solution {sol_num}: ARI={ari:.4f}, "
                  f"Source in train: {source_in_train}/{train_size}, Target in test: {target_in_test}/{test_size}")


if __name__ == "__main__":
    main()
