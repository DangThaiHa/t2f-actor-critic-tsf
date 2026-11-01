import os
import json
from typing import Dict, List

def combine_benchmark_results(base_dir: str = "benchmarks") -> Dict:
    """
    Combine all benchmark results into a single dictionary, preserving the directory structure.
    
    Args:
        base_dir (str): Base directory containing benchmark results
        
    Returns:
        Dict: Combined results with structure:
        {
            "dataset_name": {
                "l{local_length}_g{global_length}": {
                    "ep{epoch_number}": {results...}
                }
            }
        }
    """
    combined_results = {}
    
    # Walk through directory structure
    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        combined_results[dataset] = {}
        
        for length_dir in os.listdir(dataset_path):
            length_path = os.path.join(dataset_path, length_dir)
            if not os.path.isdir(length_path):
                continue
                
            combined_results[dataset][length_dir] = {}
            
            for epoch_dir in os.listdir(length_path):
                epoch_path = os.path.join(length_path, epoch_dir)
                if not os.path.isdir(epoch_path):
                    continue
                    
                results_file = os.path.join(epoch_path, 'results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        combined_results[dataset][length_dir][epoch_dir] = results
    
    return combined_results

def save_combined_results(combined_results: Dict, output_file: str = "combined_benchmarks.json"):
    """
    Save combined results to a JSON file.
    
    Args:
        combined_results (Dict): Combined benchmark results
        output_file (str): Output file path
    """
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

if __name__ == "__main__":
    # Combine all results
    results = combine_benchmark_results()
    
    # Save to file
    save_combined_results(results)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("-" * 30)
    for dataset in results:
        print(f"\nDataset: {dataset}")
        for length_config in results[dataset]:
            num_epochs = len(results[dataset][length_config])
            print(f"  {length_config}: {num_epochs} epoch configurations")