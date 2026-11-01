import logging
from io import StringIO

import numpy as np
import pandas as pd

from typing import List, Tuple, Dict

from sklearn.metrics import mean_absolute_error, mean_squared_error

def base_logger():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    log_capture_string = StringIO()
    log_capture_string.truncate(0)
    log_capture_string.seek(0)

    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add our stream handler
    logger.addHandler(ch)

    # Set the root logger level
    logger.setLevel(logging.DEBUG)

    return logger

# Helper functions
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def wsmae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return np.mean(weights * np.abs((y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)))

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    #adapt lengths if full length is not produced
    y_true = y_true[:len(y_pred)]
    weights = weights[:len(y_pred)]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_val = smape(y_true, y_pred)
    wsmae_val = wsmae(y_true, y_pred, weights)
    return {'mae': mae, 'rmse': rmse, 'smape': smape_val, 'wsmae': wsmae_val}

def prepare_data(df: pd.DataFrame, test_size: float = 0.2,
                 timestamp_col = 'timestamp',
                 series_id_col = 'series_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for time series forecasting, respecting the temporal order and series structure.
    
    Args:
    df (pd.DataFrame): Input dataframe with MultiIndex (series_id, timestamp)
    test_size (float): Proportion of data to use for testing
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
    """
    # Ensure the dataframe is sorted by series_id and timestamp
    df = df.sort_index()

    # Get unique timestamps and series_ids
    timestamps = df.index.get_level_values(timestamp_col).unique().sort_values()
    series_ids = df.index.get_level_values(series_id_col).unique()

    # Calculate the split point
    split_point = int(len(timestamps) * (1 - test_size))
    split_timestamp = timestamps[split_point]

    # Split the data
    train_df = df.loc[pd.IndexSlice[:, :split_timestamp], :]
    test_df = df.loc[pd.IndexSlice[:, split_timestamp:], :]

    # Ensure all series are represented in both train and test sets
    for series_id in series_ids:
        # Use groupby to check series existence
        train_series = train_df.groupby(level=series_id_col).groups.get(series_id, [])
        test_series = test_df.groupby(level=series_id_col).groups.get(series_id, [])
        
        if len(train_series) == 0:
            # Add earliest data points to training set
            series_data = df.groupby(level=series_id_col).get_group(series_id)
            train_df = pd.concat([train_df, series_data.iloc[:split_point]])
            
        if len(test_series) == 0:
            # Add latest data points to test set
            series_data = df.groupby(level=series_id_col).get_group(series_id)
            test_df = pd.concat([test_df, series_data.iloc[-split_point:]])

    # Sort the dataframes again to ensure proper order
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    return train_df, test_df

def infer_frequency(timestamps: pd.Series, timestamp_format: str, logger=None) -> str:
    """
    Infer the frequency of timestamps series.
    
    Parameters:
    -----------
    timestamps : pd.Series
        Series of timestamps
        
    Returns:
    --------
    str
        Inferred frequency as pandas offset string
    """
    try:
        timestamps = pd.to_datetime(timestamps, format=timestamp_format)

        # Get unique time differences
        unique_diffs = pd.Series(timestamps.sort_values().diff().dropna().unique())
        
        # Get the most common difference
        most_common_diff = unique_diffs.value_counts().index[0]
        
        # Convert timedelta to frequency string
        days = most_common_diff.days
        hours = most_common_diff.seconds // 3600
        minutes = (most_common_diff.seconds % 3600) // 60
        
        if days >= 1:
            return 'D' if days == 1 else f'{days}D'
        elif hours >= 1:
            return 'H' if hours == 1 else f'{hours}H'
        elif minutes >= 1:
            return 'T' if minutes == 1 else f'{minutes}T'
        else:
            return 'S'  # Default to seconds if shorter
    except Exception as e:
        logger.warning(f"Failed to infer frequency: {str(e)}. Using default 'D'")
        return 'D'

def pre_input(
    df: pd.DataFrame,
    series_id_col: str,
    timestamp_col: str,
    timestamp_format: str,
    covariate_cols: List[str],
    weight_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Prepare tabular data for use with T2f model.
    Ensures MultiIndex level names persist through operations.
    """
    try:
        # Initial validations and conversions remain the same
        required_cols = [series_id_col, timestamp_col, weight_col, target_col] + covariate_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df_prep = df.copy()
        df_prep[series_id_col] = df_prep[series_id_col].astype(str)
        
        try:
            df_prep[timestamp_col] = pd.to_datetime(df_prep[timestamp_col], format=timestamp_format)
        except Exception as e:
            raise ValueError(f"Failed to convert timestamps using format {timestamp_format}: {str(e)}")

        # Validate numeric data
        numeric_cols = covariate_cols + [target_col, weight_col]
        for col in numeric_cols:
            try:
                df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')
            except Exception as e:
                raise ValueError(f"Failed to convert column {col} to numeric: {str(e)}")

        # Create DataFrame
        result_df = pd.DataFrame(df_prep)
        
        # Sort by series_id and timestamp before setting index
        result_df = result_df.sort_values([series_id_col, timestamp_col])
        
        # Set index with explicit names
        result_df = result_df.set_index([series_id_col, timestamp_col])
 
        # Ensure index names are set
        result_df.index.set_names([series_id_col, timestamp_col], inplace=True)
        
        return result_df

    except Exception as e:
        raise ValueError(f"Data preparation failed: {str(e)}")
    