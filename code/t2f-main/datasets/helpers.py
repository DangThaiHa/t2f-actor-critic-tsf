from joblib import Parallel, delayed
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from joblib import dump, Parallel, delayed
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from typing import List, Dict
import warnings
import py7zr
import tempfile
import os
import pathlib
warnings.filterwarnings('ignore')
    
def extract_7z(filepath: str) -> str:
    """
    Extract 7z file and return path to extracted file.
    Returns original path if file is not 7z.
    """
    if not filepath.endswith('.7z'):
        return filepath
        
    # Create temp directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract archive
        with py7zr.SevenZipFile(filepath, mode='r') as z:
            z.extractall(temp_dir)
        
        # Get the extracted file path (assume single file in archive)
        extracted_files = list(pathlib.Path(temp_dir).glob('*'))
        if not extracted_files:
            raise ValueError(f"No files found in 7z archive: {filepath}")
            
        # Copy extracted file to a new temporary file that will persist
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extracted_files[0].suffix)
        temp_file.close()
        os.replace(str(extracted_files[0]), temp_file.name)
        
        return temp_file.name

def adf_t(col_series):    
    """Minimal ADF test returning only stationarity boolean"""
    try:
        return adfuller(col_series, autolag='AIC')[1] <= 0.05
    except:
        return True

def process_local_series(group_data: pd.DataFrame, 
                        v_cols: List[str],
                        non_value_cols: List[str],
                        target_col: str,
                        md_ord: int = 6) -> pd.DataFrame:
    """Process a single local series - optimized for parallel execution"""
    df_l_i = group_data[non_value_cols].copy()
    df_l_d = group_data[v_cols].copy()
    
    diff_order = 0
    while diff_order < md_ord:
        if all(adf_t(df_l_d[col]) for col in v_cols):
            break
        df_l_d = df_l_d.diff().dropna()
        diff_order += 1
    
    df_l = pd.concat([df_l_i, df_l_d], axis=1)
    df_l['diff_order'] = diff_order
    return df_l.reindex(columns=[col for col in df_l.columns if col != target_col] + [target_col])

def ds_prep(
        dir: str,
        fname: str,
        fform: str,
        ts_index: str,
        l_label: str,
        variate_cols: list[str],
        cont_factors: list[str],
        target_col: str,
        timestamp_format: str = '%Y-%m-%d',
        n_jobs: int = -1,  # Use all available cores by default
        cleanup_temp: bool = True,  # Whether to cleanup temporary files
    ):
    # Handle potential 7z compression
    original_fpath = dir + fname + fform
    if original_fpath.endswith('.7z'):
        fpath = extract_7z(original_fpath)
        fform = pathlib.Path(fpath).suffix  # Get the actual file format after extraction
    else:
        fpath = original_fpath
    
    try:
        # Read file efficiently
        read_func = pd.read_csv if '.csv' in fform.lower() else pd.read_parquet
        df0 = read_func(fpath)
        
        # Optimize memory usage by converting to appropriate dtypes
        for col in df0.select_dtypes(include=['object']).columns:
            if col != ts_index:  # Skip timestamp column
                df0[col] = pd.Categorical(df0[col])
        
        # Drop duplicates early to reduce processing
        df0.drop_duplicates(inplace=True)
        
        # Process timestamp once
        df0[ts_index] = pd.to_datetime(df0[ts_index], format=timestamp_format)
        df0.sort_values(ts_index, ascending=True, inplace=True)
        
        # Prepare column lists once
        v_cols = list(set(cont_factors + [target_col]))
        non_value_cols = [col for col in df0.columns if col not in v_cols]
        
        # Group data for parallel processing
        groups = [group for _, group in df0.groupby(l_label)]

        # Create a progress bar
        total_groups = len(groups)

        # Process groups in parallel without embedding progress bar in workers
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_local_series)(
                group,
                v_cols,
                non_value_cols,
                target_col
            ) for group in tqdm(groups, desc="Processing", file=sys.stdout)
        )

        # Combine results efficiently
        df_diff = pd.concat(results, ignore_index=True)
        df_diff.dropna(inplace=True)
        df_diff.drop_duplicates(inplace=True)
        
        # Save results efficiently
        output_fname = 'diff_' + fname
        if '.csv' in fform.lower():
            df_diff.to_csv(output_fname + '.csv', index=False)
        else:
            df_diff.to_parquet(
                output_fname + '.parquet',
                index=False,
                compression='snappy'
            )
        
        # Efficient scaling
        scale_cols = [c for c in df_diff.columns if c != l_label]
        scale_data = df_diff[scale_cols].set_index(ts_index).to_numpy()
        
        scaler = StandardScaler(copy=False)
        scaler.fit(scale_data)
        
        # Save artifacts efficiently
        dump(scaler, 'scaler_' + fname + '.joblib', compress=3)
        np.save('scaler_stats_' + fname + '.npy', 
                np.array([scaler.mean_, scaler.scale_]), 
                allow_pickle=False)

    finally:
        # Cleanup temporary files if needed
        if cleanup_temp and fpath != original_fpath:
            try:
                os.unlink(fpath)
            except:
                pass