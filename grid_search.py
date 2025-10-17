"""
Grid Search for Neural Network Hyperparameter Tuning
Runs experiments and logs results to a single file without creating directories
"""

import os
import json
import time
import subprocess
import pandas as pd
from itertools import product


def create_grid_parameters():
    """
    Define the hyperparameter grid for experimentation.
    """
    return {
        'pca_components': [20, 25, 30, 35, 40],
        'lr': [0.001, 0.01, 0.02, 0.03],
        'hidden_layers': [
            [128, 64], 
            [256, 128, 64],
            [512, 256, 128],
            [512, 256, 128, 64]
        ],
        'dropout': [0.0, 0.1],
        'batch_size': [32, 64, 96]
    }


def build_command_line_args(experiment_params):
    """
    Build command line arguments for main.py.
    Uses a single output directory to avoid creating multiple folders.
    """
    cmd = [
        "python", "main.py",
        "--pca_components", str(experiment_params['pca_components']),
        "--lr", str(experiment_params['lr']),
        "--hidden_layers"
    ]
    
    for layer in experiment_params['hidden_layers']:
        cmd.append(str(layer))
    
    cmd.extend([
        "--dropout", str(experiment_params['dropout']),
        "--batch_size", str(experiment_params['batch_size']),
        "--output_dir", "grid_output",
        "--epochs", "500",
        "--patience", "20"
    ])
    
    return cmd


def extract_metrics_from_file():
    """
    Extract metrics from JSON file.
    """
    metrics_file = "grid_output/metrics.json"
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            print(f"Error reading metrics file: {e}")
            return {}
    else:
        print(f"Metrics file not found: {metrics_file}")
        return {}


def run_single_experiment(cmd, experiment_id, params):
    """
    Run a single experiment and capture metrics from output.
    """
    print(f"Full command: {' '.join(cmd)}")
    print(f"Starting experiment {experiment_id}")
    print(f"Parameters: PCA={params['pca_components']}, LR={params['lr']}, "
          f"Layers={params['hidden_layers']}, Dropout={params['dropout']}, "
          f"Batch={params['batch_size']}")
    
    start_time = time.time()
    
    try:
        # Run the experiment with timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=3600
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            # Extract metrics from json file
            metrics = extract_metrics_from_file()
            
            if metrics:
                status = "SUCCESS"
                print(f"Experiment {experiment_id} completed - R²: {metrics.get('r2', 'N/A'):.4f}")
            else:
                status = "NO_METRICS"
                metrics = {}
                print(f"Experiment {experiment_id} completed but no metrics found")
        else:
            status = "FAILED"
            metrics = {}
            print(f"Experiment {experiment_id} failed")
            
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        execution_time = 3600
        metrics = {}
        print(f"Experiment {experiment_id} timed out")
    except Exception as e:
        status = "ERROR"
        execution_time = time.time() - start_time
        metrics = {}
        print(f"Experiment {experiment_id} error: {e}")
    
    # Build experiment result
    experiment_result = {
        'experiment_id': experiment_id,
        'status': status,
        'execution_time': execution_time,
        'pca_components': params['pca_components'],
        'lr': params['lr'],
        'hidden_layers': str(params['hidden_layers']),
        'dropout': params['dropout'],
        'batch_size': params['batch_size'],
        'command': ' '.join(cmd)
    }
    
    # Add metrics if available
    experiment_result.update(metrics)
    
    return experiment_result


def save_results_to_file(results, filename='grid_search_results.txt'):
    """
    Save all results to a single text file with ranking.
    """
    # Filter successful experiments
    successful_experiments = [r for r in results if r['status'] == 'SUCCESS' and 'r2' in r]
    
    # Sort by R² score (descending)
    successful_experiments.sort(key=lambda x: x['r2'], reverse=True)
    
    with open(filename, 'w') as f:
        f.write("GRID SEARCH RESULTS - RANKED BY R² SCORE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Successful experiments: {len(successful_experiments)}\n")
        f.write(f"Completion date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write ranked results
        f.write("RANKED RESULTS:\n")
        f.write("-" * 80 + "\n")
        
        for rank, exp in enumerate(successful_experiments, 1):
            f.write(f"Rank {rank} (R²: {exp['r2']:.4f}):\n")
            f.write(f"  Experiment ID: {exp['experiment_id']}\n")
            f.write(f"  PCA Components: {exp['pca_components']}\n")
            f.write(f"  Learning Rate: {exp['lr']}\n")
            f.write(f"  Hidden Layers: {exp['hidden_layers']}\n")
            f.write(f"  Dropout: {exp['dropout']}\n")
            f.write(f"  Batch Size: {exp['batch_size']}\n")
            f.write(f"  RMSE: {exp.get('rmse', 'N/A')}\n")
            f.write(f"  MAE: {exp.get('mae', 'N/A')}\n")
            f.write(f"  Execution Time: {exp['execution_time']:.1f}s\n")
            f.write(f"  Command: {exp['command']}\n")
            f.write("-" * 80 + "\n")
        
        # Write failed experiments summary
        failed_experiments = [r for r in results if r['status'] != 'SUCCESS']
        if failed_experiments:
            f.write("\nFAILED EXPERIMENTS:\n")
            f.write("-" * 80 + "\n")
            for exp in failed_experiments:
                f.write(f"ID {exp['experiment_id']}: {exp['status']} - {exp['command']}\n")


def main():
    """Main function to run grid search"""
    print("Starting Neural Network Grid Search")
    print("Results will be saved to grid_search_results.txt")
    
    # Create single output directory
    os.makedirs("grid_output", exist_ok=True)
    
    # Get parameter grid
    param_grid = create_grid_parameters()
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))
    
    print(f"Total parameter combinations: {len(param_combinations)}")
    
    results = []
    
    # Run experiments sequentially
    for i, param_values in enumerate(param_combinations, 1):
        # Create parameter dictionary
        params_dict = dict(zip(param_names, param_values))
        
        # Build command line arguments
        cmd = build_command_line_args(params_dict)
        
        # Run experiment
        result = run_single_experiment(cmd, i, params_dict)
        results.append(result)
        
        # Save progress after each experiment
        save_results_to_file(results)
        
        # Small delay between experiments
        time.sleep(1)
    
    print("Grid search completed! Check grid_search_results.txt for ranked results")


if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    print(f"Main.py exists: {os.path.exists('main.py')}")
    main()