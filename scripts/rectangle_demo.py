import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chart_patterns.plotting import display_chart_pattern
from chart_patterns.rectangles import find_rectangle_pattern


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, default="./data/eurusd-4h.csv")
    parser.add_argument("--start", required=False, type=int, default=0)
    parser.add_argument("--end", required=False, type=int, default=0)
    parser.add_argument("--out", required=False, default="rectangle_demo")
    parser.add_argument("--lookback", required=False, type=int, default=80)
    parser.add_argument("--tolerance", required=False, type=float, default=0.0008)
    parser.add_argument("--max_images", required=False, type=int, default=10)
    args = parser.parse_args()

    ohlc = pd.read_csv(args.csv)
    if args.end and args.end > args.start:
        ohlc = ohlc.iloc[args.start : args.end, :]
    ohlc = ohlc.reset_index(drop=True)

    ohlc = find_rectangle_pattern(
        ohlc,
        lookback=int(args.lookback),
        min_touches_per_side=2,
        tolerance=float(args.tolerance),
        progress=True,
    )

    rect_points = ohlc.index[ohlc["chart_type"] == "rectangle"].tolist()
    if rect_points:
        keep = set(rect_points[: int(args.max_images)])
        ohlc = ohlc.loc[ohlc.index.isin(keep) | (ohlc.index >= min(keep) - 150)].reset_index(drop=True)

    display_chart_pattern(ohlc, pattern="rectangle", save=True, lookback=120, image_subdir=str(args.out))


if __name__ == "__main__":
    main()
