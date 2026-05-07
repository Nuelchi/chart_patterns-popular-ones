import numpy as np
import pandas as pd

from .pivot_points import find_all_pivot_points


def find_rectangle_pattern(
    ohlc: pd.DataFrame,
    lookback: int = 60,
    min_touches_per_side: int = 2,
    tolerance: float = 0.0005,
    progress: bool = False,
) -> pd.DataFrame:
    ohlc["chart_type"] = ohlc.get("chart_type", "")

    ohlc["rectangle_high"] = np.nan
    ohlc["rectangle_low"] = np.nan
    ohlc["rectangle_high_idx"] = [np.array([]) for _ in range(len(ohlc))]
    ohlc["rectangle_low_idx"] = [np.array([]) for _ in range(len(ohlc))]
    ohlc["rectangle_point"] = np.nan

    ohlc = find_all_pivot_points(ohlc)

    candle_iter = range(lookback, len(ohlc))
    if progress:
        from tqdm import tqdm

        candle_iter = tqdm(candle_iter, desc="Finding rectangle patterns")

    for candle_idx in candle_iter:
        window = range(candle_idx - lookback, candle_idx + 1)

        pivot_high_idxs = []
        pivot_high_vals = []
        pivot_low_idxs = []
        pivot_low_vals = []

        for i in window:
            if ohlc.loc[i, "pivot"] == 2:
                pivot_high_idxs.append(i)
                pivot_high_vals.append(float(ohlc.loc[i, "high"]))
            elif ohlc.loc[i, "pivot"] == 1:
                pivot_low_idxs.append(i)
                pivot_low_vals.append(float(ohlc.loc[i, "low"]))

        if len(pivot_high_idxs) < min_touches_per_side or len(pivot_low_idxs) < min_touches_per_side:
            continue

        high_level = float(np.median(pivot_high_vals))
        low_level = float(np.median(pivot_low_vals))
        if not (high_level > low_level):
            continue

        try:
            if candle_idx > 0 and str(ohlc.loc[candle_idx - 1, "chart_type"]) == "rectangle":
                prev_high = ohlc.loc[candle_idx - 1, "rectangle_high"]
                prev_low = ohlc.loc[candle_idx - 1, "rectangle_low"]
                if (
                    not pd.isna(prev_high)
                    and not pd.isna(prev_low)
                    and abs(float(prev_high) - high_level) <= tolerance
                    and abs(float(prev_low) - low_level) <= tolerance
                ):
                    continue
        except Exception:
            pass

        high_touch_mask = [abs(v - high_level) <= tolerance for v in pivot_high_vals]
        low_touch_mask = [abs(v - low_level) <= tolerance for v in pivot_low_vals]

        high_touch_idxs = np.array([idx for idx, ok in zip(pivot_high_idxs, high_touch_mask) if ok], dtype=float)
        low_touch_idxs = np.array([idx for idx, ok in zip(pivot_low_idxs, low_touch_mask) if ok], dtype=float)

        if len(high_touch_idxs) < min_touches_per_side or len(low_touch_idxs) < min_touches_per_side:
            continue

        ohlc.loc[candle_idx, "chart_type"] = "rectangle"
        ohlc.loc[candle_idx, "rectangle_high"] = high_level
        ohlc.loc[candle_idx, "rectangle_low"] = low_level
        ohlc.at[candle_idx, "rectangle_high_idx"] = high_touch_idxs
        ohlc.at[candle_idx, "rectangle_low_idx"] = low_touch_idxs
        ohlc.loc[candle_idx, "rectangle_point"] = candle_idx

    return ohlc
