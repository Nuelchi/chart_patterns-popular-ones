import numpy as np
import pandas as pd

from .pivot_points import find_all_pivot_points
from .utils import check_ohlc_names


def find_n_shape_pattern(
    ohlc: pd.DataFrame,
    lookback: int = 250,
    bos_ratio: float = 1.0,
    min_hl_ratio: float = 0.0,
    min_bd_bars: int = 3,
    min_bd_atr: float = 0.8,
    min_ac_bars: int = 3,
    min_ac_atr: float = 0.6,
    retest_lookahead: int = 250,
    progress: bool = False,
) -> pd.DataFrame:
    ohlc = ohlc.copy()
    check_ohlc_names(ohlc)

    ohlc["chart_type"] = ""
    ohlc["nshape_dir"] = ""
    ohlc["nshape_idx"] = [np.array([]) for _ in range(len(ohlc))]
    ohlc["nshape_point"] = [np.array([]) for _ in range(len(ohlc))]
    ohlc["nshape_entry_level"] = np.nan
    ohlc["nshape_entry_idx"] = np.nan

    ohlc = find_all_pivot_points(ohlc)

    tr = (ohlc["high"].astype(float) - ohlc["low"].astype(float)).abs()
    ohlc["_atr14"] = tr.rolling(14).mean()

    if not progress:
        candle_iter = range(max(int(lookback), 10), len(ohlc))
    else:
        try:
            from tqdm import tqdm

            candle_iter = tqdm(range(max(int(lookback), 10), len(ohlc)), desc="Finding N-shape patterns...")
        except Exception:
            candle_iter = range(max(int(lookback), 10), len(ohlc))

    for i in candle_iter:
        window = ohlc.iloc[max(0, i - int(lookback)) : i + 1]
        piv = window[window["pivot"] != 0]
        if piv.empty:
            continue

        pivot_indices = piv.index.to_list()
        if pivot_indices and pivot_indices[-1] == i:
            pivot_indices = pivot_indices[:-1]

        last_high_idx = -1
        last_low_idx = -1
        for idx in reversed(pivot_indices):
            if last_high_idx == -1 and int(ohlc.loc[idx, "pivot"]) == 1:
                last_high_idx = int(idx)
            if last_low_idx == -1 and int(ohlc.loc[idx, "pivot"]) == 2:
                last_low_idx = int(idx)
            if last_high_idx != -1 and last_low_idx != -1:
                break

        # Bullish N
        if last_high_idx != -1:
            b_idx = int(last_high_idx)
            b_level = float(ohlc.loc[b_idx, "high"])

            a_idx = -1
            for idx in reversed([p for p in pivot_indices if int(p) < b_idx]):
                if int(ohlc.loc[int(idx), "pivot"]) == 2:
                    a_idx = int(idx)
                    break

            if a_idx != -1:
                a_level = float(ohlc.loc[a_idx, "low"])
                atr_ref = float(ohlc.loc[b_idx, "_atr14"]) if not pd.isna(ohlc.loc[b_idx, "_atr14"]) else float(tr.iloc[max(0, b_idx - 14) : b_idx + 1].mean())

                d_idx = -1
                for j in range(b_idx + 1, len(ohlc)):
                    if (j - b_idx) < int(min_bd_bars):
                        continue
                    close_j = float(ohlc.loc[j, "close"])
                    if close_j > b_level * float(bos_ratio) and (close_j - b_level) >= (atr_ref * float(min_bd_atr)):
                        d_idx = int(j)
                        break

                if d_idx != -1 and d_idx > b_idx + 1:
                    seg = ohlc.loc[b_idx + 1 : d_idx - 1, :]
                    if not seg.empty:
                        seg_piv = seg[seg["pivot"] == 2]
                        if not seg_piv.empty:
                            c_idx = int(seg_piv["low"].astype(float).idxmin())
                        else:
                            c_idx = int(seg["low"].astype(float).idxmin())
                        c_level = float(ohlc.loc[c_idx, "low"])

                        if c_level >= a_level * (1.0 + float(min_hl_ratio)):
                            if (c_idx - a_idx) < int(min_ac_bars):
                                pass
                            else:
                                atr_a = float(ohlc.loc[a_idx, "_atr14"]) if not pd.isna(ohlc.loc[a_idx, "_atr14"]) else float(tr.iloc[max(0, a_idx - 14) : a_idx + 1].mean())
                                if (c_level - a_level) >= (atr_a * float(min_ac_atr)):
                                    e_idx = -1
                                    end_scan = min(len(ohlc) - 1, d_idx + int(retest_lookahead))
                                    for k in range(d_idx + 1, end_scan + 1):
                                        if float(ohlc.loc[k, "low"]) <= a_level:
                                            e_idx = int(k)
                                            break

                                    if e_idx != -1:
                                        ohlc.loc[d_idx, "chart_type"] = "nshape"
                                        ohlc.loc[d_idx, "nshape_dir"] = "bull"
                                        ohlc.at[d_idx, "nshape_idx"] = np.array([a_idx, b_idx, c_idx, d_idx, e_idx], dtype=int)
                                        ohlc.at[d_idx, "nshape_point"] = np.array([
                                            float(ohlc.loc[a_idx, "low"]),
                                            float(ohlc.loc[b_idx, "high"]),
                                            float(ohlc.loc[c_idx, "low"]),
                                            float(ohlc.loc[d_idx, "high"]),
                                            float(a_level),
                                        ], dtype=float)
                                        ohlc.loc[d_idx, "nshape_entry_level"] = float(a_level)
                                        ohlc.loc[d_idx, "nshape_entry_idx"] = float(e_idx)

        # Bearish (inverse) N
        if last_low_idx != -1:
            b_idx = int(last_low_idx)
            b_level = float(ohlc.loc[b_idx, "low"])

            a_idx = -1
            for idx in reversed([p for p in pivot_indices if int(p) < b_idx]):
                if int(ohlc.loc[int(idx), "pivot"]) == 1:
                    a_idx = int(idx)
                    break

            if a_idx != -1:
                a_level = float(ohlc.loc[a_idx, "high"])
                atr_ref = float(ohlc.loc[b_idx, "_atr14"]) if not pd.isna(ohlc.loc[b_idx, "_atr14"]) else float(tr.iloc[max(0, b_idx - 14) : b_idx + 1].mean())

                d_idx = -1
                for j in range(b_idx + 1, len(ohlc)):
                    if (j - b_idx) < int(min_bd_bars):
                        continue
                    close_j = float(ohlc.loc[j, "close"])
                    if close_j < (b_level / float(bos_ratio)) and (b_level - close_j) >= (atr_ref * float(min_bd_atr)):
                        d_idx = int(j)
                        break

                if d_idx != -1 and d_idx > b_idx + 1:
                    seg = ohlc.loc[b_idx + 1 : d_idx - 1, :]
                    if not seg.empty:
                        seg_piv = seg[seg["pivot"] == 1]
                        if not seg_piv.empty:
                            c_idx = int(seg_piv["high"].astype(float).idxmax())
                        else:
                            c_idx = int(seg["high"].astype(float).idxmax())
                        c_level = float(ohlc.loc[c_idx, "high"])

                        if c_level <= a_level / (1.0 + float(min_hl_ratio)):
                            if (c_idx - a_idx) < int(min_ac_bars):
                                pass
                            else:
                                atr_a = float(ohlc.loc[a_idx, "_atr14"]) if not pd.isna(ohlc.loc[a_idx, "_atr14"]) else float(tr.iloc[max(0, a_idx - 14) : a_idx + 1].mean())
                                if (a_level - c_level) >= (atr_a * float(min_ac_atr)):
                                    e_idx = -1
                                    end_scan = min(len(ohlc) - 1, d_idx + int(retest_lookahead))
                                    for k in range(d_idx + 1, end_scan + 1):
                                        if float(ohlc.loc[k, "high"]) >= a_level:
                                            e_idx = int(k)
                                            break

                                    if e_idx != -1:
                                        ohlc.loc[d_idx, "chart_type"] = "nshape"
                                        ohlc.loc[d_idx, "nshape_dir"] = "bear"
                                        ohlc.at[d_idx, "nshape_idx"] = np.array([a_idx, b_idx, c_idx, d_idx, e_idx], dtype=int)
                                        ohlc.at[d_idx, "nshape_point"] = np.array([
                                            float(ohlc.loc[a_idx, "high"]),
                                            float(ohlc.loc[b_idx, "low"]),
                                            float(ohlc.loc[c_idx, "high"]),
                                            float(ohlc.loc[d_idx, "low"]),
                                            float(a_level),
                                        ], dtype=float)
                                        ohlc.loc[d_idx, "nshape_entry_level"] = float(a_level)
                                        ohlc.loc[d_idx, "nshape_entry_idx"] = float(e_idx)

    if "_atr14" in ohlc.columns:
        ohlc = ohlc.drop(columns=["_atr14"])

    return ohlc
