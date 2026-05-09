import numpy as np
import pandas as pd

from .pivot_points import find_all_pivot_points


def find_liquidity_sweep_pattern(
    ohlc: pd.DataFrame,
    lookback: int = 80,
    buffer: float = 0.0,
    progress: bool = False,
) -> pd.DataFrame:
    """Detect liquidity sweeps (bull + bear) using pivot highs/lows.

    Bull sweep:
      - price wicks below a prior pivot low (low < level - buffer)
      - closes back above the level (close > level)

    Bear sweep:
      - price wicks above a prior pivot high (high > level + buffer)
      - closes back below the level (close < level)

    The function marks the sweep candle as chart_type='liq_sweep' and stores plotting fields.
    """

    ohlc["chart_type"] = ohlc.get("chart_type", "")

    ohlc["liq_sweep_type"] = ""
    ohlc["liq_sweep_level"] = np.nan
    ohlc["liq_sweep_pivot_idx"] = -1
    ohlc["liq_sweep_sweep_idx"] = -1
    ohlc["liq_sweep_zone_low"] = np.nan
    ohlc["liq_sweep_zone_high"] = np.nan

    ohlc = find_all_pivot_points(ohlc)

    candle_iter = range(lookback, len(ohlc))
    if progress:
        from tqdm import tqdm

        candle_iter = tqdm(candle_iter, desc="Finding liquidity sweeps")

    for candle_idx in candle_iter:
        window = range(candle_idx - lookback, candle_idx)

        pivot_high_idxs = []
        pivot_low_idxs = []

        for i in window:
            if ohlc.loc[i, "pivot"] == 2:
                pivot_high_idxs.append(i)
            elif ohlc.loc[i, "pivot"] == 1:
                pivot_low_idxs.append(i)

        if not pivot_high_idxs and not pivot_low_idxs:
            continue

        high = float(ohlc.loc[candle_idx, "high"])
        low = float(ohlc.loc[candle_idx, "low"])
        close = float(ohlc.loc[candle_idx, "close"])

        # Prefer the most recent pivot on each side.
        pivot_high_idx = int(pivot_high_idxs[-1]) if pivot_high_idxs else -1
        pivot_low_idx = int(pivot_low_idxs[-1]) if pivot_low_idxs else -1

        pivot_high_level = float(ohlc.loc[pivot_high_idx, "high"]) if pivot_high_idx != -1 else np.nan
        pivot_low_level = float(ohlc.loc[pivot_low_idx, "low"]) if pivot_low_idx != -1 else np.nan

        # Bear sweep (buy-side liquidity): wick above pivot high then close back below.
        bear_ok = (
            pivot_high_idx != -1
            and not pd.isna(pivot_high_level)
            and high > (pivot_high_level + float(buffer))
            and close < pivot_high_level
        )

        # Bull sweep (sell-side liquidity): wick below pivot low then close back above.
        bull_ok = (
            pivot_low_idx != -1
            and not pd.isna(pivot_low_level)
            and low < (pivot_low_level - float(buffer))
            and close > pivot_low_level
        )

        if not (bear_ok or bull_ok):
            continue

        # Avoid spamming identical consecutive detections.
        try:
            if candle_idx > 0 and str(ohlc.loc[candle_idx - 1, "chart_type"]) == "liq_sweep":
                prev_type = str(ohlc.loc[candle_idx - 1, "liq_sweep_type"])
                prev_level = ohlc.loc[candle_idx - 1, "liq_sweep_level"]
                if (
                    prev_type
                    and not pd.isna(prev_level)
                    and ((bear_ok and prev_type == "bear" and abs(float(prev_level) - pivot_high_level) <= buffer) or
                         (bull_ok and prev_type == "bull" and abs(float(prev_level) - pivot_low_level) <= buffer))
                ):
                    continue
        except Exception:
            pass

        if bear_ok and (not bull_ok or pivot_high_idx >= pivot_low_idx):
            level = float(pivot_high_level)
            ohlc.loc[candle_idx, "chart_type"] = "liq_sweep"
            ohlc.loc[candle_idx, "liq_sweep_type"] = "bear"
            ohlc.loc[candle_idx, "liq_sweep_level"] = level
            ohlc.loc[candle_idx, "liq_sweep_pivot_idx"] = pivot_high_idx
            ohlc.loc[candle_idx, "liq_sweep_sweep_idx"] = candle_idx
            ohlc.loc[candle_idx, "liq_sweep_zone_low"] = level
            ohlc.loc[candle_idx, "liq_sweep_zone_high"] = high
        else:
            level = float(pivot_low_level)
            ohlc.loc[candle_idx, "chart_type"] = "liq_sweep"
            ohlc.loc[candle_idx, "liq_sweep_type"] = "bull"
            ohlc.loc[candle_idx, "liq_sweep_level"] = level
            ohlc.loc[candle_idx, "liq_sweep_pivot_idx"] = pivot_low_idx
            ohlc.loc[candle_idx, "liq_sweep_sweep_idx"] = candle_idx
            ohlc.loc[candle_idx, "liq_sweep_zone_low"] = low
            ohlc.loc[candle_idx, "liq_sweep_zone_high"] = level

    return ohlc
