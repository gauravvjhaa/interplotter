import os
import pandas as pd
import numpy as np

# Path to your dataset directory
dataset_dir = "datasets"

# Ensure the directory exists
if not os.path.isdir(dataset_dir):
    raise FileNotFoundError(f"Directory not found: {dataset_dir}")

# Process all CSV files
for filename in os.listdir(dataset_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(dataset_dir, filename)
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Identify columns with any NaN or 0
        cols_to_drop = df.columns[df.isnull().any() | (df == 0).any()]
        
        # Drop the columns
        cleaned_df = df.drop(columns=cols_to_drop)
        
        # Save back to the same file (overwrite)
        cleaned_df.to_csv(file_path, index=False)
        
        print(f"Cleaned: {filename} | Removed columns: {list(cols_to_drop)}")
