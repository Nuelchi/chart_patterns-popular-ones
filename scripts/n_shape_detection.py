import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chart_patterns.pivot_points import find_all_pivot_points


def _load_ohlc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    lower_cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_cols)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required OHLC columns: {sorted(missing)}")
    return df.reset_index(drop=True)


def detect_n_shape(
    ohlc: pd.DataFrame,
    lookback: int = 250,
    bos_ratio: float = 1.0,
    min_hl_ratio: float = 0.0,
    min_bd_bars: int = 3,
    min_bd_atr: float = 0.8,
    min_ac_bars: int = 3,
    min_ac_atr: float = 0.6,
    retest_lookahead: int = 250,
    min_gap: int = 30,
) -> pd.DataFrame:
    ohlc = ohlc.copy()
    ohlc["n_dir"] = ""
    ohlc["n_A_idx"] = np.nan
    ohlc["n_A_level"] = np.nan
    ohlc["n_B_idx"] = np.nan
    ohlc["n_B_level"] = np.nan
    ohlc["n_C_idx"] = np.nan
    ohlc["n_C_level"] = np.nan
    ohlc["n_D_idx"] = np.nan
    ohlc["n_D_level"] = np.nan
    ohlc["n_E_idx"] = np.nan
    ohlc["n_E_level"] = np.nan

    ohlc = find_all_pivot_points(ohlc)
    tr = (ohlc["high"].astype(float) - ohlc["low"].astype(float)).abs()
    ohlc["_atr14"] = tr.rolling(14).mean()

    i = max(lookback, 10)
    while i < len(ohlc):
        window = ohlc.iloc[max(0, i - lookback) : i + 1]
        piv = window[window["pivot"] != 0]
        if piv.empty:
            i += 1
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

        if last_high_idx != -1:
            b_idx = last_high_idx
            b_level = float(ohlc.loc[b_idx, "high"])
            a_idx = -1
            for idx in reversed([p for p in pivot_indices if int(p) < b_idx]):
                if int(ohlc.loc[int(idx), "pivot"]) == 2:
                    a_idx = int(idx)
                    break
            if a_idx != -1:
                a_level = float(ohlc.loc[a_idx, "low"])
                d_idx = -1
                atr_ref = float(ohlc.loc[b_idx, "_atr14"]) if not pd.isna(ohlc.loc[b_idx, "_atr14"]) else float(tr.iloc[max(0, b_idx - 14) : b_idx + 1].mean())
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
                            # Enforce that the pullback leg (A->C) is not tiny/clustered.
                            # - Bars: C must be at least N bars after A
                            # - Price: (C - A) must be at least ATR * min_ac_atr
                            if (c_idx - a_idx) < int(min_ac_bars):
                                i += 1
                                continue
                            atr_a = float(ohlc.loc[a_idx, "_atr14"]) if not pd.isna(ohlc.loc[a_idx, "_atr14"]) else float(tr.iloc[max(0, a_idx - 14) : a_idx + 1].mean())
                            if (c_level - a_level) < (atr_a * float(min_ac_atr)):
                                i += 1
                                continue
                            e_idx = -1
                            end_scan = min(len(ohlc) - 1, d_idx + int(retest_lookahead))
                            for k in range(d_idx + 1, end_scan + 1):
                                if float(ohlc.loc[k, "low"]) <= a_level:
                                    e_idx = int(k)
                                    break
                            if e_idx != -1:
                                ohlc.loc[d_idx, "n_dir"] = "bull"
                                ohlc.loc[d_idx, "n_A_idx"] = float(a_idx)
                                ohlc.loc[d_idx, "n_A_level"] = float(a_level)
                                ohlc.loc[d_idx, "n_B_idx"] = float(b_idx)
                                ohlc.loc[d_idx, "n_B_level"] = float(b_level)
                                ohlc.loc[d_idx, "n_C_idx"] = float(c_idx)
                                ohlc.loc[d_idx, "n_C_level"] = float(c_level)
                                ohlc.loc[d_idx, "n_D_idx"] = float(d_idx)
                                ohlc.loc[d_idx, "n_D_level"] = float(b_level)
                                ohlc.loc[d_idx, "n_E_idx"] = float(e_idx)
                                ohlc.loc[d_idx, "n_E_level"] = float(a_level)
                                i = e_idx + int(min_gap)
                                continue

        if last_low_idx != -1:
            b_idx = last_low_idx
            b_level = float(ohlc.loc[b_idx, "low"])
            a_idx = -1
            for idx in reversed([p for p in pivot_indices if int(p) < b_idx]):
                if int(ohlc.loc[int(idx), "pivot"]) == 1:
                    a_idx = int(idx)
                    break
            if a_idx != -1:
                a_level = float(ohlc.loc[a_idx, "high"])
                d_idx = -1
                atr_ref = float(ohlc.loc[b_idx, "_atr14"]) if not pd.isna(ohlc.loc[b_idx, "_atr14"]) else float(tr.iloc[max(0, b_idx - 14) : b_idx + 1].mean())
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
                                i += 1
                                continue
                            atr_a = float(ohlc.loc[a_idx, "_atr14"]) if not pd.isna(ohlc.loc[a_idx, "_atr14"]) else float(tr.iloc[max(0, a_idx - 14) : a_idx + 1].mean())
                            if (a_level - c_level) < (atr_a * float(min_ac_atr)):
                                i += 1
                                continue
                            e_idx = -1
                            end_scan = min(len(ohlc) - 1, d_idx + int(retest_lookahead))
                            for k in range(d_idx + 1, end_scan + 1):
                                if float(ohlc.loc[k, "high"]) >= a_level:
                                    e_idx = int(k)
                                    break
                            if e_idx != -1:
                                ohlc.loc[d_idx, "n_dir"] = "bear"
                                ohlc.loc[d_idx, "n_A_idx"] = float(a_idx)
                                ohlc.loc[d_idx, "n_A_level"] = float(a_level)
                                ohlc.loc[d_idx, "n_B_idx"] = float(b_idx)
                                ohlc.loc[d_idx, "n_B_level"] = float(b_level)
                                ohlc.loc[d_idx, "n_C_idx"] = float(c_idx)
                                ohlc.loc[d_idx, "n_C_level"] = float(c_level)
                                ohlc.loc[d_idx, "n_D_idx"] = float(d_idx)
                                ohlc.loc[d_idx, "n_D_level"] = float(b_level)
                                ohlc.loc[d_idx, "n_E_idx"] = float(e_idx)
                                ohlc.loc[d_idx, "n_E_level"] = float(a_level)
                                i = e_idx + int(min_gap)
                                continue

        i += 1

    if "_atr14" in ohlc.columns:
        ohlc = ohlc.drop(columns=["_atr14"]) 
    return ohlc


def _plot_cycle(
    ohlc: pd.DataFrame,
    row_idx: int,
    out_png: Path,
    out_html: Path,
    title: str,
    left_buffer: int,
    right_buffer: int,
) -> bool:
    if row_idx < 0 or row_idx >= len(ohlc):
        return False
    row = ohlc.iloc[int(row_idx)]
    if str(row.get("n_dir", "")) == "":
        return False

    a_idx = int(row["n_A_idx"]) if not pd.isna(row["n_A_idx"]) else -1
    b_idx = int(row["n_B_idx"]) if not pd.isna(row["n_B_idx"]) else -1
    c_idx = int(row["n_C_idx"]) if not pd.isna(row["n_C_idx"]) else -1
    d_idx = int(row["n_D_idx"]) if not pd.isna(row["n_D_idx"]) else int(row_idx)
    e_idx = int(row["n_E_idx"]) if not pd.isna(row["n_E_idx"]) else -1
    if min(a_idx, b_idx, c_idx, d_idx, e_idx) < 0:
        return False

    entry_level = float(row["n_A_level"]) if not pd.isna(row["n_A_level"]) else np.nan

    start = max(0, a_idx - int(left_buffer))
    end = min(len(ohlc) - 1, e_idx + int(right_buffer))
    view = ohlc.iloc[start : end + 1].copy()
    view_index = list(range(start, end + 1))

    avg_range = float((view["high"].astype(float) - view["low"].astype(float)).rolling(14).mean().iloc[-1])
    if not np.isfinite(avg_range) or avg_range <= 0:
        avg_range = float((view["high"].astype(float) - view["low"].astype(float)).mean())
    zone_h = avg_range * 0.22

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=view_index,
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            increasing_line_color="#2ECC71",
            decreasing_line_color="#E74C3C",
        )
    )

    if not pd.isna(entry_level):
        fig.add_shape(type="line", x0=start, x1=end, y0=entry_level, y1=entry_level, line=dict(color="white", width=2))
        fig.add_shape(
            type="rect",
            x0=max(start, a_idx - 2),
            x1=min(end, a_idx + 6),
            y0=entry_level - zone_h,
            y1=entry_level + zone_h,
            fillcolor="#F1C40F",
            opacity=0.35,
            line=dict(color="#F1C40F", width=0),
            layer="below",
        )

    is_bull = str(row["n_dir"]) == "bull"
    a_y = float(ohlc.loc[a_idx, "low"]) if is_bull else float(ohlc.loc[a_idx, "high"])
    b_y = float(ohlc.loc[b_idx, "high"]) if is_bull else float(ohlc.loc[b_idx, "low"])
    c_y = float(ohlc.loc[c_idx, "low"]) if is_bull else float(ohlc.loc[c_idx, "high"])
    d_y = float(ohlc.loc[d_idx, "high"]) if is_bull else float(ohlc.loc[d_idx, "low"])

    # N connector: straight segments with a slanted mid-leg.
    # We draw A->B->C->D as a polyline, and add tiny chamfers around B and C
    # so the intersections read like an N (no spline curvature, no blocky right angles).
    base_x = [float(a_idx), float(b_idx), float(c_idx), float(d_idx)]
    base_y = [float(a_y), float(b_y), float(c_y), float(d_y)]

    chamfer = 0.14
    pts_x = [base_x[0]]
    pts_y = [base_y[0]]
    for i in range(1, len(base_x) - 1):
        px, py = base_x[i - 1], base_y[i - 1]
        cx, cy = base_x[i], base_y[i]
        nx, ny = base_x[i + 1], base_y[i + 1]

        bx1 = cx + (px - cx) * chamfer
        by1 = cy + (py - cy) * chamfer
        bx2 = cx + (nx - cx) * chamfer
        by2 = cy + (ny - cy) * chamfer

        pts_x.extend([bx1, cx, bx2])
        pts_y.extend([by1, cy, by2])

    pts_x.append(base_x[-1])
    pts_y.append(base_y[-1])

    fig.add_trace(
        go.Scatter(
            x=pts_x,
            y=pts_y,
            mode="lines",
            line=dict(color="#00D1FF", width=8),
            name="N",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[a_idx, b_idx, c_idx, d_idx],
            y=[a_y, b_y, c_y, d_y],
            mode="markers",
            marker=dict(size=11, color="#00D1FF"),
            name=None,
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[e_idx],
            y=[entry_level],
            mode="markers",
            marker=dict(size=14, color="#F1C40F", symbol="square"),
            name="E",
        )
    )

    fig.add_annotation(x=a_idx, y=a_y, text="A", showarrow=False, yshift=-18, font=dict(color="#FFFFFF", size=18))
    fig.add_annotation(x=b_idx, y=b_y, text="B", showarrow=False, yshift=18, font=dict(color="#FFFFFF", size=18))
    fig.add_annotation(x=c_idx, y=c_y, text="C", showarrow=False, yshift=-18, font=dict(color="#FFFFFF", size=18))
    fig.add_annotation(x=d_idx, y=d_y, text="D", showarrow=False, yshift=18, font=dict(color="#FFFFFF", size=18))
    fig.add_annotation(x=e_idx, y=entry_level, text="E", showarrow=False, yshift=-22, font=dict(color="#F1C40F", size=18))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        width=1180,
        height=680,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_png))
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="nshape")
    p.add_argument("--direction", choices=["bull", "bear", "both"], default="bull")
    p.add_argument("--lookback", type=int, default=250)
    p.add_argument("--bos-ratio", type=float, default=1.0)
    p.add_argument("--min-hl-ratio", type=float, default=0.0)
    p.add_argument("--min-bd-bars", type=int, default=3, help="Minimum bars between B and D")
    p.add_argument("--min-bd-atr", type=float, default=0.8, help="Minimum ATR(14) distance for BOS beyond B")
    p.add_argument("--min-ac-bars", type=int, default=3, help="Minimum bars between A and C")
    p.add_argument("--min-ac-atr", type=float, default=0.6, help="Minimum ATR(14) distance between A and C")
    p.add_argument("--retest-lookahead", type=int, default=250)
    p.add_argument("--min-gap", type=int, default=30)
    p.add_argument("--left-buffer", type=int, default=40)
    p.add_argument("--right-buffer", type=int, default=40)
    p.add_argument("--max-plots", type=int, default=40)
    args = p.parse_args()

    ohlc = _load_ohlc(Path(args.csv))
    ohlc = detect_n_shape(
        ohlc,
        lookback=int(args.lookback),
        bos_ratio=float(args.bos_ratio),
        min_hl_ratio=float(args.min_hl_ratio),
        min_bd_bars=int(args.min_bd_bars),
        min_bd_atr=float(args.min_bd_atr),
        min_ac_bars=int(args.min_ac_bars),
        min_ac_atr=float(args.min_ac_atr),
        retest_lookahead=int(args.retest_lookahead),
        min_gap=int(args.min_gap),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = ohlc["n_dir"] != ""
    if args.direction != "both":
        mask = mask & (ohlc["n_dir"] == args.direction)
    det_idx = ohlc.index[mask].to_list()
    if not det_idx:
        raise SystemExit("No N-shape detections found.")

    saved = 0
    for ridx in det_idx:
        if saved >= int(args.max_plots):
            break
        png_path = out_dir / f"{args.prefix}_{saved:03d}_idx{int(ridx)}.png"
        html_path = out_dir / f"{args.prefix}_{saved:03d}_idx{int(ridx)}.html"
        ok = _plot_cycle(
            ohlc,
            int(ridx),
            png_path,
            html_path,
            title=f"N-shape ({str(ohlc.loc[int(ridx), 'n_dir'])})",
            left_buffer=int(args.left_buffer),
            right_buffer=int(args.right_buffer),
        )
        if ok:
            saved += 1

    if saved == 0:
        raise SystemExit("Detections existed but no plots could be generated.")


if __name__ == "__main__":
    main()
