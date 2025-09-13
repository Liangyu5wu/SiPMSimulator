#!/usr/bin/env python3
"""Merge parallel parameter sweep results"""
import pandas as pd
import glob
import os

def main():
    # Find all job result files
    files = sorted(glob.glob('param_sweep_results_job_*.csv'))
    
    if not files:
        print("No job result files found")
        return
    
    print(f"Found {len(files)} job files")
    
    # Read and merge all results
    all_dfs = []
    total_experiments = 0
    
    for file in files:
        df = pd.read_csv(file)
        if len(df) > 0:
            all_dfs.append(df)
            total_experiments += len(df)
            print(f"  {file}: {len(df)} experiments")
    
    if not all_dfs:
        print("All files are empty")
        return
    
    # Merge and sort by validation loss
    merged = pd.concat(all_dfs, ignore_index=True).sort_values('val_loss')
    
    # Save merged results
    output_file = 'param_sweep_results.csv'
    merged.to_csv(output_file, index=False)
    
    # Remove individual job files
    for file in files:
        os.remove(file)
    
    print(f"\nMerge completed!")
    print(f"Total experiments: {total_experiments}")
    print(f"Results saved to: {output_file}")
    print(f"Removed {len(files)} job files")
    
    # Show best results
    print(f"\nTop 5 configurations:")
    cols = ['lr', 'batch_size', 'filters', 'kernels', 'val_loss', 'correlation']
    print(merged.head()[cols].to_string(index=False))

if __name__ == "__main__":
    main()