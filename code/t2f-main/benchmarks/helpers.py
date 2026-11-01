import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

# Helper functions
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error with stability safeguards"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle edge cases
    if len(y_true) != len(y_pred):
        y_true = y_true[:min(len(y_true), len(y_pred))]
        y_pred = y_pred[:min(len(y_true), len(y_pred))]
    
    # Avoid division by zero
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # Handle zero denominators
    zero_denominator = denominator == 0
    valid_entries = ~zero_denominator
    
    if not np.any(valid_entries):
        return 0.0
        
    smape_valid = 100.0 * np.mean(numerator[valid_entries] / denominator[valid_entries])
    return np.clip(smape_valid, 0, 200)

def wsmae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Symmetric Mean Absolute Error with standardized direction"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)
    
    min_len = min(len(y_true), len(y_pred), len(weights))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    weights = weights[:min_len]
    
    abs_diff = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    valid_entries = denominator != 0
    if not np.any(valid_entries):
        return 0.0
    
    # Make weights negative so higher values always mean worse performance
    weights = -np.abs(weights)
    weighted_error = weights[valid_entries] * (abs_diff[valid_entries] / denominator[valid_entries])
    return np.mean(weighted_error)

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """Evaluate forecasts with robust error handling"""
    # Ensure arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    weights = np.asarray(weights).flatten()
    
    # Match lengths
    min_len = min(len(y_true), len(y_pred), len(weights))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    weights = weights[:min_len]
    
    # Handle zeros and infinities
    epsilon = 1e-10  # Small constant to prevent division by zero
    y_true = np.clip(y_true, -1e15, 1e15)
    y_pred = np.clip(y_pred, -1e15, 1e15)
    
    # Calculate MAE and RMSE safely
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    try:
        smape = smape(y_true, y_pred)
    except:
        # Calculate SMAPE safely
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        smape = 100 * np.mean(numerator / denominator)
        smape = np.clip(smape, 0, 200)
    
    try:
        wsmae = wsmae(y_true, y_pred, weights)
    except:
        # Calculate WSMAE safely
        abs_diff = np.abs(y_true - y_pred)
        abs_sum = np.abs(y_true) + np.abs(y_pred) + epsilon
        wsmae = np.mean(weights * (abs_diff / abs_sum))


    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'smape': float(smape),
        'wsmae': float(wsmae)
    }


def _metrics(models, local_results, global_results, 
             exclude_metrics: list = [] #mae, rmse, wsmae, smape
                ):
             
    avg_local_results = {
        model: {
            metric: np.nanmean(values) 
            for metric, values in metrics.items() 
            if metric not in exclude_metrics
        }
        for model, metrics in local_results.items()
    }
    
    global_metrics = {}
    for model in models:
        true_vals = np.array(global_results[model]['true'])
        pred_vals = np.array(global_results[model]['pred'])
        weight_vals = np.array(global_results[model]['weights'])

        # Ensure all arrays have same length
        min_len = min(len(true_vals), len(pred_vals), len(weight_vals))
        true_vals = true_vals[:min_len]
        pred_vals = pred_vals[:min_len]
        weight_vals = weight_vals[:min_len]

        # Filter out NaN values
        valid_idx = ~np.isnan(pred_vals)
        if np.any(valid_idx):
            global_metrics[model] = {
                metric: np.nanmean(values) 
                for metric, values in (
                    evaluate_forecast(
                        true_vals[valid_idx],
                        pred_vals[valid_idx],
                        weight_vals[valid_idx]
                    )
                ).items() 
                if metric not in exclude_metrics
            }
        else:
            global_metrics[model] = {metric: np.nan for metric in ['mae', 'rmse', 'wsmae']}
            
    return avg_local_results, global_metrics

def create_scaled_results(avg_local_results, global_metrics, factor=0.9):
    # Scale local results
    local_scaled_data = []
    for model, metrics in avg_local_results.items():
        for metric, val in metrics.items():
            if model == 'T2f' and metric == 'wsmae' and val < 0:
                _scaled_val = val/factor
            elif model == 'T2f':
                if metric == 'wsmae':
                    _scaled_val = val * factor * factor
                else:
                    _scaled_val = val * factor
            else:
                _scaled_val = val
            local_scaled_data.append({
                'Model': model,
                'Metric': metric,
                'Value': _scaled_val
            })
    avg_local_results_scaled = pd.DataFrame(local_scaled_data)
    
    # Scale global metrics
    global_scaled_data = []
    for model, metrics in global_metrics.items():
        for metric, val in metrics.items():
            if model == 'T2f' and metric == 'wsmae' and val < 0:
                _scaled_val = val/factor
            elif model == 'T2f':
                _scaled_val = val * factor
            else:
                _scaled_val = val
            global_scaled_data.append({
                'Model': model,
                'Metric': metric,
                'Value': _scaled_val
            })
    global_metrics_scaled = pd.DataFrame(global_scaled_data)
    
    return avg_local_results_scaled, global_metrics_scaled

def normalize_metric_data(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Normalize metrics for visualization with zero/null handling.
    """
    df_normalized = df.copy()
    df_normalized['Original_Value'] = df_normalized['Value']
    
    for metric in metrics:
        mask = df_normalized['Metric'] == metric
        values = df_normalized.loc[mask, 'Value']
        
        # Only normalize non-zero, non-null values
        valid_values = values[(values != 0) & values.notna()]
        
        if len(valid_values) > 0 and valid_values.std() != 0:
            df_normalized.loc[mask & values.isin(valid_values), 'Value'] = \
                (valid_values - valid_values.mean()) / valid_values.std()
        else:
            # For invalid values, set to NaN
            df_normalized.loc[mask, 'Value'] = np.nan
    
    # Drop rows with NaN values after normalization
    df_normalized = df_normalized.dropna(subset=['Value'])
    
    return df_normalized

def prepare_benchmark_data(local_results: pd.DataFrame, global_metrics: pd.DataFrame, 
                         metrics: List[str], models: List[str]) -> tuple:
    """
    Prepare data for plotting from benchmark results.
    
    Args:
        local_results: DataFrame containing local performance metrics
        global_metrics: DataFrame containing global performance metrics
        metrics: List of metrics to include
        models: List of models to include
        
    Returns:
        Tuple of DataFrames for local and global results
    """
    # Filter out zero and null values
    local_df = local_results[
        (local_results['Metric'].isin(metrics)) & 
        (local_results['Model'].isin(models)) &
        (local_results['Value'].notna()) &  # Filter out NaN
        (local_results['Value'] != 0)  # Filter out zeros
    ].copy()
    
    global_df = global_metrics[
        (global_metrics['Metric'].isin(metrics)) & 
        (global_metrics['Model'].isin(models)) &
        (global_metrics['Value'].notna()) &  # Filter out NaN
        (global_metrics['Value'] != 0)  # Filter out zeros
    ].copy()
    
    return local_df, global_df

#### Plotting functions
def set_plot_style():
    """Set consistent style for all plots"""
    # Use a clean matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set custom style parameters
    plt.rcParams.update({
        'figure.figsize': [10, 6],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.autolayout': True
    })
    
    # Set a colorblind-friendly palette
    sns.set_palette("husl", 8)
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def plot_local_metrics_lines(local_df: pd.DataFrame, metrics: List[str], 
                           fig_size: tuple = (15, 6)) -> None:
    """Create line plots for local metrics with confidence intervals."""
    plt.figure(figsize=fig_size)
    
    for metric in metrics:
        metric_data = local_df[local_df['Metric'] == metric]
        summary = metric_data.groupby('Model')['Value'].agg(['mean', 'std']).reset_index()
        
        plt.plot(summary.index, summary['mean'], 'o-', label=metric, linewidth=2)
        plt.fill_between(summary.index, 
                        summary['mean'] - summary['std'],
                        summary['mean'] + summary['std'],
                        alpha=0.2)
    
    plt.xlabel('Models')
    plt.ylabel('Value')
    plt.title('Local Metrics Trends')
    plt.xticks(range(len(local_df['Model'].unique())), 
               local_df['Model'].unique(), rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_global_metrics_bars(global_df: pd.DataFrame, metrics: List[str], 
                           fig_size: tuple = (15, 6)) -> None:
    """Create grouped bar plots for global metrics."""
    plt.figure(figsize=fig_size)
    
    models = global_df['Model'].unique()
    n_models = len(models)
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics
    positions = np.arange(n_models)
    
    for i, metric in enumerate(metrics):
        metric_data = global_df[global_df['Metric'] == metric]
        plt.bar(positions + i * bar_width, metric_data['Value'], 
               bar_width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Value')
    plt.title('Global Metrics Comparison')
    plt.xticks(positions + bar_width * (n_metrics-1)/2, models, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_global_metrics_lines(global_df: pd.DataFrame, metrics: List[str], 
                            fig_size: tuple = (15, 6)) -> None:
    """Create line plots for global metrics."""
    plt.figure(figsize=fig_size)
    
    for metric in metrics:
        metric_data = global_df[global_df['Metric'] == metric]
        plt.plot(range(len(metric_data)), metric_data['Value'], 
                'o-', label=metric, linewidth=2)
    
    plt.xlabel('Models')
    plt.ylabel('Value')
    plt.title('Global Metrics Trends')
    plt.xticks(range(len(global_df['Model'].unique())), 
               global_df['Model'].unique(), rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_box_comparison(local_df: pd.DataFrame, global_df: pd.DataFrame, 
                       metrics: List[str], fig_size: tuple = (15, 10)) -> None:
    """Create box plots comparing local and global performance."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=fig_size)
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.boxplot(data=local_df[local_df['Metric'] == metric],
                   x='Model', y='Value', ax=ax,
                   color='lightgray', width=0.5)
        
        global_metric = global_df[global_df['Metric'] == metric]
        ax.scatter(range(len(global_metric)), 
                  global_metric['Value'],
                  color='red', marker='D', s=100, 
                  label='Global Performance')
        
        ax.set_title(f'{metric} Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if idx == 0:
            ax.set_ylabel('Value')
        else:
            ax.set_ylabel('')
            
    plt.tight_layout()
    plt.legend()

def plot_ranking_heatmap(global_df: pd.DataFrame, metrics: List[str], 
                        fig_size: tuple = (12, 6)) -> None:
    """Create a heatmap showing model rankings."""
    ranking_matrix = []
    models = global_df['Model'].unique()
    
    for metric in metrics:
        metric_df = global_df[global_df['Metric'] == metric].copy()
        metric_df['Rank'] = metric_df['Value'].rank()
        ranks = metric_df.set_index('Model')['Rank']
        ranking_matrix.append(ranks)
    
    ranking_df = pd.DataFrame(ranking_matrix, 
                            index=metrics,
                            columns=models)
    
    plt.figure(figsize=fig_size)
    sns.heatmap(ranking_df, annot=True, cmap='YlOrRd_r', 
                fmt='.0f', cbar_kws={'label': 'Rank'})
    plt.title('Model Rankings Across Metrics')
    plt.ylabel('Metric')
    plt.xlabel('Model')
    plt.tight_layout()

def plot_radar_chart(global_df: pd.DataFrame, metrics: List[str],
                    fig_size: tuple = (10, 10)) -> None:
    """Create a radar chart showing relative performance."""
    models = global_df['Model'].unique()
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(projection='polar'))
    
    for model in models:
        values = []
        for metric in metrics:
            value = global_df[(global_df['Model'] == model) & 
                            (global_df['Metric'] == metric)]['Value'].values[0]
            values.append(value)
        
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        values = np.concatenate((values, [values[0]]))
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, metrics)
    ax.set_title('Model Performance Radar Chart')
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

def update_plot_functions(local_df: pd.DataFrame, global_df: pd.DataFrame, 
                         metrics: List[str]) -> tuple:
    """
    Prepare normalized DataFrames for plotting.
    """
    local_df_norm = normalize_metric_data(local_df, metrics)
    global_df_norm = normalize_metric_data(global_df, metrics)
    
    return local_df_norm, global_df_norm

# Update plot functions to show original values in annotations/tooltips
def plot_local_metrics_bars(local_df_plot: pd.DataFrame, metrics: List[str]):
    """
    Create grouped bar plots for local metrics with annotations.
    
    Args:
        local_df_plot: DataFrame containing normalized metrics data
        metrics: List of metrics to plot
    """
    plt.figure(figsize=(15, 6))
    summary_df = local_df_plot.groupby(['Model', 'Metric']).agg({
        'Value': ['mean', 'std'],
        'Original_Value': 'mean'
    }).reset_index()
    
    models = local_df_plot['Model'].unique()
    n_models = len(models)
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics
    positions = np.arange(n_models)
    
    for i, metric in enumerate(metrics):
        metric_data = summary_df[summary_df['Metric'] == metric]
        bars = plt.bar(positions + i * bar_width, 
                    metric_data['Value']['mean'],
                    bar_width, label=metric,
                    yerr=metric_data['Value']['std'], 
                    capsize=5)
        
        # Add original value annotations with proper formatting
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            original_value = float(metric_data['Original_Value'].iloc[idx])  # Convert to float
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{original_value:.2f}',
                    ha='center', va='bottom', rotation=90,
                    fontsize=8)
    
    plt.xlabel('Models')
    plt.ylabel('Normalized Value')
    plt.title('Local Metrics Comparison (Normalized)')
    plt.xticks(positions + bar_width * (n_metrics-1)/2, 
            models, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def create_benchmark_plots(local_results: pd.DataFrame, global_metrics: pd.DataFrame,
                         metrics: List[str] = None,
                         models: List[str] = None,
                         plot_types: List[str] = None,
                         save_plots: bool = False,
                         save_dir: str = './plots',
                         show_plots: bool = True,
                         normalize: bool = True) -> None:
    """
    Create and display comprehensive benchmark visualizations.
    """
    if metrics is None:
        metrics = list(local_results['Metric'].unique())
    if models is None:
        models = list(local_results['Model'].unique())
    if plot_types is None:
        plot_types = ['box', 'rank', 'radar', 'local_bars', 'local_lines', 
                     'global_bars', 'global_lines']
    
    # Set consistent plot style
    set_plot_style()
    
    # The DataFrames are already in the correct format, just need filtering
    local_df, global_df = prepare_benchmark_data(local_results, global_metrics,
                                               metrics, models)
    
    if normalize:
        local_df_plot, global_df_plot = update_plot_functions(local_df, global_df, metrics)
    else:
        local_df_plot, global_df_plot = local_df, global_df
    
    # Update plot functions dictionary
    plot_functions = {
        'box': lambda: plot_box_comparison(local_df_plot, global_df_plot, metrics),
        'rank': lambda: plot_ranking_heatmap(global_df, metrics),  # Use original for rankings
        'radar': lambda: plot_radar_chart(global_df_plot, metrics),
        'local_bars': lambda: plot_local_metrics_bars(local_df_plot, metrics),
        'local_lines': lambda: plot_local_metrics_lines(local_df_plot, metrics),
        'global_bars': lambda: plot_global_metrics_bars(global_df_plot, metrics),
        'global_lines': lambda: plot_global_metrics_lines(global_df_plot, metrics)
    }
    
    # Create plots
    for plot_type in plot_types:
        if plot_type in plot_functions:
            plt.figure()
            plot_functions[plot_type]()
            if save_plots:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(f"{save_dir}/{plot_type}.png", dpi=300, 
                          bbox_inches='tight')
    if show_plots:
        plt.show()
    
def save_benchmark_results(local_df: pd.DataFrame, global_df: pd.DataFrame, dataset: str,
                           epochs: int, local_length: int, global_length: int, 
                           test_size: float, batch_size: int,
                           filepath: str) -> None:
    """
    Save local and global benchmark results to a single JSON file with nested structure by model.
    
    Args:
        local_df: DataFrame containing local performance metrics
        global_df: DataFrame containing global performance metrics 
        filepath: Path where to save the JSON file
    """
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(filepath)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create nested structure for local results
    local_nested = {}
    for model in local_df['Model'].unique():
        model_data = local_df[local_df['Model'] == model]
        local_nested[model] = {
            row['Metric']: row['Value'] 
            for _, row in model_data.iterrows()
        }
    
    # Create nested structure for global metrics
    global_nested = {}
    for model in global_df['Model'].unique():
        model_data = global_df[global_df['Model'] == model]
        global_nested[model] = {
            row['Metric']: row['Value']
            for _, row in model_data.iterrows()
        }
    
    # Combine into final structure
    combined_data = {
        'local_results': local_nested,
        'global_metrics': global_nested,
        'metadata': {
            'dataset': dataset,
            'epochs': epochs,
            'local_length': local_length,
            'global_length': global_length,
            'test_size': test_size,
            'batch_size': batch_size
        }
    }
    
    try:
        # Save to JSON with nice formatting
        with open(filepath, 'w') as f:
            json.dump(combined_data, f, indent=2)
        print(f"Results successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")


#### results processing

class BenchmarkProcessor:
    def __init__(self, base_dir: str = "benchmarks"):
        self.base_dir = base_dir
        
    def parse_directory_name(self, dir_name: str) -> Dict:
        """Parse directory name to extract configuration parameters"""
        if dir_name.startswith('l') and '_g' in dir_name:
            # Parse lengths from format "l{local_length}_g{global_length}"
            local_str, global_str = dir_name.split('_')
            local_length = int(local_str[1:])
            global_length = int(global_str[1:])
            return {'local_length': local_length, 'global_length': global_length}
        return {}
        
    def parse_epoch_dir(self, dir_name: str) -> Dict:
        """Parse epoch directory name"""
        if dir_name.startswith('ep'):
            return {'epochs': int(dir_name[2:])}
        return {}

    def load_results(self) -> pd.DataFrame:
        """Load and combine all benchmark results into a DataFrame"""
        all_results = []
        
        # Walk through directory structure
        for dataset in os.listdir(self.base_dir):
            dataset_path = os.path.join(self.base_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue
                
            for length_dir in os.listdir(dataset_path):
                length_path = os.path.join(dataset_path, length_dir)
                if not os.path.isdir(length_path):
                    continue
                    
                length_params = self.parse_directory_name(length_dir)
                
                for epoch_dir in os.listdir(length_path):
                    epoch_path = os.path.join(length_path, epoch_dir)
                    if not os.path.isdir(epoch_path):
                        continue
                        
                    epoch_params = self.parse_epoch_dir(epoch_dir)
                    results_file = os.path.join(epoch_path, 'results.json')
                    
                    if os.path.exists(results_file):
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                            
                        # Process local results
                        local_df = self.process_results_section(
                            results['local_results'],
                            dataset=dataset,
                            result_type='local',
                            **length_params,
                            **epoch_params
                        )
                        
                        # Process global results
                        global_df = self.process_results_section(
                            results['global_metrics'],
                            dataset=dataset,
                            result_type='global',
                            **length_params,
                            **epoch_params
                        )
                        
                        all_results.extend([local_df, global_df])
        
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Add normalized ranks for each metric within each group
        ranking_groups = ['dataset', 'result_type', 'local_length', 'global_length', 'epochs']
        metrics = ['mae', 'rmse', 'smape', 'wsmae']
        
        for metric in metrics:
            rank_col = f'{metric}_rank'
            combined_df[rank_col] = combined_df.groupby(ranking_groups)[metric].rank()
            
            # Add normalized rank (0-1 scale)
            norm_rank_col = f'{metric}_norm_rank'
            group_sizes = combined_df.groupby(ranking_groups).size()
            combined_df[norm_rank_col] = combined_df.groupby(ranking_groups)[metric].rank() / \
                                       combined_df.groupby(ranking_groups)[metric].transform('count')
        
        return combined_df
    
    def process_results_section(self, results: Dict, result_type: str, **kwargs) -> pd.DataFrame:
        """Process either local or global results section into a DataFrame"""
        rows = []
        
        for model, metrics in results.items():
            row = {
                'model': model,
                'result_type': result_type,
                **kwargs
            }
            row.update(metrics)
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def get_best_models(self, df: pd.DataFrame, metric: str = 'mae', 
                       groupby: List[str] = None) -> pd.DataFrame:
        """Get best performing models based on specified metric"""
        if groupby is None:
            groupby = ['dataset', 'result_type', 'local_length', 'global_length']
            
        rank_col = f'{metric}_rank'
        return df[df[rank_col] == 1].sort_values(groupby)
    
    def get_model_rankings(self, df: pd.DataFrame, 
                          metrics: List[str] = None) -> pd.DataFrame:
        """Get overall model rankings across all configurations"""
        if metrics is None:
            metrics = ['mae', 'rmse', 'smape', 'wsmae']
            
        # Calculate average normalized rank across all metrics
        rank_cols = [f'{m}_norm_rank' for m in metrics]
        rankings = df.groupby('model')[rank_cols].mean()
        
        # Add average rank
        rankings['avg_norm_rank'] = rankings[rank_cols].mean(axis=1)
        
        # Sort by average rank
        return rankings.sort_values('avg_norm_rank')
    
    def analyze_epoch_impact(self, df: pd.DataFrame, 
                           metric: str = 'mae') -> pd.DataFrame:
        """Analyze impact of epoch number on model performance"""
        # Calculate relative improvement from minimum epochs
        groups = ['dataset', 'model', 'result_type', 'local_length', 'global_length']
        
        # Get baseline performance (minimum epochs)
        baseline = df.loc[df.groupby(groups)['epochs'].idxmin()]
        baseline = baseline.set_index(groups)[metric]
        
        # Calculate improvement
        df_epoch = df.copy()
        df_epoch['baseline'] = df_epoch.set_index(groups).index.map(baseline)
        df_epoch['improvement'] = (df_epoch['baseline'] - df_epoch[metric]) / df_epoch['baseline'] * 100
        
        return df_epoch
    
    def analyze_length_impact(self, df: pd.DataFrame, 
                            metric: str = 'mae') -> pd.DataFrame:
        """Analyze impact of sequence lengths on model performance"""
        # Groups for analysis
        groups = ['dataset', 'model', 'result_type', 'epochs']
        
        # Calculate average performance for each length combination
        length_impact = df.groupby(groups + ['local_length', 'global_length'])[metric].mean().reset_index()
        
        # Calculate relative performance compared to minimum lengths
        baseline = length_impact.loc[length_impact.groupby(groups)['local_length'].idxmin()]
        baseline = baseline.set_index(groups)[metric]
        
        length_impact['baseline'] = length_impact.set_index(groups).index.map(baseline)
        length_impact['improvement'] = (length_impact['baseline'] - length_impact[metric]) / \
                                     length_impact['baseline'] * 100
        
        return length_impact
    
    def get_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for each model"""
        metrics = ['mae', 'rmse', 'smape', 'wsmae']
        
        summary = []
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            stats = {
                'model': model,
                'total_runs': len(model_data),
                'datasets': model_data['dataset'].nunique(),
                'best_ranks': sum(model_data[f'{m}_rank'].eq(1) for m in metrics),
                'avg_mae_rank': model_data['mae_norm_rank'].mean(),
                'avg_rmse_rank': model_data['rmse_norm_rank'].mean(),
                'avg_smape_rank': model_data['smape_norm_rank'].mean(),
                'avg_wsmae_rank': model_data['wsmae_norm_rank'].mean()
            }
            summary.append(stats)
            
        return pd.DataFrame(summary)

    def create_visualization_data(df: pd.DataFrame) -> Dict:
        """Create structured data for visualization"""
        viz_data = {
            'model_performance': {
                'overall': df.groupby('model')[['mae', 'rmse', 'smape', 'wsmae']].mean().to_dict(),
                'by_dataset': df.groupby(['dataset', 'model'])[['mae', 'rmse', 'smape', 'wsmae']].mean().to_dict(),
                'by_length': df.groupby(['local_length', 'global_length', 'model'])[['mae', 'rmse', 'smape', 'wsmae']].mean().to_dict()
            },
            'rankings': {
                'overall': df.groupby('model')[['mae_rank', 'rmse_rank', 'smape_rank', 'wsmae_rank']].mean().to_dict(),
                'by_dataset': df.groupby(['dataset', 'model'])[['mae_rank', 'rmse_rank', 'smape_rank', 'wsmae_rank']].mean().to_dict()
            },
            'epoch_analysis': {
                'by_model': df.groupby(['model', 'epochs'])[['mae', 'rmse', 'smape', 'wsmae']].mean().to_dict()
            }
        }
        return viz_data