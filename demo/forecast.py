"""Giải thích dự báo Stacking trên tập test ETTm, chân trời 30 bước."""
import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / 'results/final/full/ETTm_h30'


def explain(data, weights):
    baseline = data['last'].astype(float)
    contributions = {
        model: weight * (data[model].astype(float) - baseline)
        for model, weight in weights.items()
    }
    reconstructed = baseline + sum(contributions.values())
    return contributions, reconstructed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', nargs='?', default='final', choices=['final'])
    parser.add_argument('--station', choices=['1', '2'], default='1')
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--step', type=int, choices=range(1, 31), default=30)
    parser.add_argument('--verify', action='store_true', help='Kiểm tra mọi dự báo test')
    args = parser.parse_args()
    weights = json.loads((RESULTS / 'selection.json').read_text())['weights']
    np.testing.assert_allclose(sum(weights.values()), 1, atol=1e-8)
    with np.load(RESULTS / 'forecasts.npz') as data:
        contributions, predictions = explain(data, weights)
        if args.verify:
            np.testing.assert_allclose(predictions, data['stacked'], atol=1e-4, rtol=0)
            error = np.max(np.abs(predictions - data['stacked']))
            print(f'{predictions.size} dự báo; sai lệch tối đa {error:.9f}')
            return
        rows = np.flatnonzero(data['series'].astype(str) == args.station)
        rows = rows[np.argsort(data['origin'][rows])]
        if not 1 <= args.window <= len(rows):
            parser.error(f'Cửa sổ hợp lệ từ 1 đến {len(rows)}')
        row, step = rows[args.window - 1], args.step - 1
        origin = data['origin'][row].astype('datetime64[m]')
        target = origin + np.timedelta64(15 * args.step, 'm')
        print(f'ETTm{args.station} | {origin} → {target} | bước {args.step}')
        print(f"OT tại gốc: {data['last'][row, step]:.3f}")
        for model, values in contributions.items():
            print(f'{model:15} dự báo {data[model][row, step]:8.3f}   đóng góp {values[row, step]:+8.3f}')
        actual = data['actual'][row, step]
        pred = predictions[row, step]
        print(f'Stacking: {pred:.3f} | OT thực: {actual:.3f} | Sai số: {abs(pred-actual):.3f}')


if __name__ == '__main__':
    main()
