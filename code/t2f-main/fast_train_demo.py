import os
import sys
import time

repo = os.getcwd()
sys.path.insert(0, repo)
sys.path.insert(0, os.path.join(repo, 't2f'))

from t2f.t2f_base import data_load, t2f_fit

dataset = 'eld'
local_length = 2
global_length = 3
ep = 2

print('Loading data...')
train_df, test_df, prepared_df, freq = data_load(
    './datasets/eld/diff_eld.parquet',
    'id',
    'timestamp',
    '%Y-%m-%d %H:%M:%S',
    ['tmax', 'tmin', 'tmed'],
    'cost_kwh',
    'load',
    0.2,
    '15min',
    id_frac=0.05,
    len_frac=0.05,
)
print(f'Data ready: train={train_df.shape}, test={test_df.shape}, freq={freq}')

start = time.time()
model = t2f_fit(
    train_df=train_df,
    test_df=test_df,
    prepared_df=prepared_df,
    freq=freq,
    series_id_col='id',
    timestamp_col='timestamp',
    covariate_cols=['tmax', 'tmin', 'tmed'],
    weight_col='cost_kwh',
    target_col='load',
    local_length=local_length,
    global_length=global_length,
    epochs=ep,
    model_save_path=f'./t2f/model-store/{dataset}/l{local_length}_g{global_length}/ep{ep}',
    hidden_size=64,
    num_layers=2,
    batch_size=16,
)
print(f'Training finished in {time.time() - start:.2f} seconds')
print(type(model).__name__)

X_test = test_df[['tmax', 'tmin', 'tmed']]
pred = model.predict(X_test)
print('Prediction shape:', pred.shape)
print(pred.head(5).to_string())
