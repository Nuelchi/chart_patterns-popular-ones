import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chart_patterns.liquidity_sweep import find_liquidity_sweep_pattern
from chart_patterns.plotting import display_chart_pattern


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, default="./data/eurusd-4h.csv")
    parser.add_argument("--start", required=False, type=int, default=0)
    parser.add_argument("--end", required=False, type=int, default=0)
    parser.add_argument("--out", required=False, default="liq_sweep_demo")
    parser.add_argument("--lookback", required=False, type=int, default=80)
    parser.add_argument("--buffer", required=False, type=float, default=0.0000)
    parser.add_argument("--max_images", required=False, type=int, default=10)
    args = parser.parse_args()

    ohlc = pd.read_csv(args.csv)
    if args.end and args.end > args.start:
        ohlc = ohlc.iloc[args.start : args.end, :]
    ohlc = ohlc.reset_index(drop=True)

    ohlc = find_liquidity_sweep_pattern(
        ohlc,
        lookback=int(args.lookback),
        buffer=float(args.buffer),
        progress=True,
    )

    sweep_points = ohlc.index[ohlc["chart_type"] == "liq_sweep"].tolist()
    if sweep_points:
        keep = set(sweep_points[: int(args.max_images)])
        ohlc = ohlc.loc[ohlc.index.isin(keep) | (ohlc.index >= min(keep) - 150)].reset_index(drop=True)

    display_chart_pattern(ohlc, pattern="liq_sweep", save=True, lookback=120, image_subdir=str(args.out))


if __name__ == "__main__":
    main()
