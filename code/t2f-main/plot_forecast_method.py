from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'datasets' / 'eld' / 'diff_eld.parquet'
OUTPUT_PATH = ROOT / 'results' / 'forecast_method_comparison.png'


def main():
    df = pd.read_parquet(DATA_PATH)
    s = df[df['id'] == 'MT_012'].sort_values('timestamp').copy()
    s['timestamp'] = pd.to_datetime(s['timestamp'])

    hist = s[s['timestamp'] < pd.Timestamp('2011-01-19 19:45:00')].tail(40)
    future = s[(s['timestamp'] >= pd.Timestamp('2011-01-19 19:45:00')) &
               (s['timestamp'] <= pd.Timestamp('2011-01-19 20:00:00'))].copy()

    pred_ts = pd.to_datetime([
        '2011-01-19 19:45:00',
        '2011-01-19 20:00:00',
    ])
    pred_vals = [0.0003021759098951679, -0.0022302400552603532]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist['timestamp'], hist['load'], label='Dữ liệu lịch sử thực tế', color='tab:blue', linewidth=2)
    ax.plot(future['timestamp'], future['load'], label='Giá trị thực tế tại điểm dự báo', color='tab:orange', marker='o', linestyle='--', linewidth=2)
    ax.plot(pred_ts, pred_vals, label='Dự đoán của mô hình T2F', color='tab:red', marker='x', linestyle='-', linewidth=2, markersize=8)

    ax.set_title('So sánh dữ liệu thực tế và dự đoán T2F cho series MT_012')
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('Load')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
