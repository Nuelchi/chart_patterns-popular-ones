"""
Date  : 2023-12-25
Author: Zetra Team

For slider: https://community.plotly.com/t/multiple-traces-with-a-single-slider-in-plotly/16356/2
"""

import os
import pandas as pd
import plotly.graph_objects as go
import sys
import numpy as np


from .utils import check_ohlc_names
from plotly.subplots import make_subplots
from tqdm import tqdm
from typing import Dict, List, Union


def set_theme(fig: go.Candlestick, theme: Dict[str,str] = {"bg_color": "black", "up_color":"#3D9970", 
                                                      "down_color": "#FF4136", "legend_font_color": "white", "xaxes_color": "white", "yaxes_color": "white"}) -> go.Candlestick:
    """
    Set the aesthetics of the plot
    
    :params fig is the candlestick object
    :type :go.Candlestick
    
    :params theme is dictionary with the settings 
    :type :Dict[str, str]
    
    :return (go.Candlestick)
    """
    
    fig.update_layout(xaxis_rangeslider_visible=False, plot_bgcolor=theme["bg_color"], paper_bgcolor=theme["bg_color"],
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, side="right"), legend_font_color=theme["legend_font_color"],
                    legend=dict(yanchor="bottom", y=0.99, xanchor="left", x=0.01) )

    fig.update_traces(increasing_fillcolor=theme["up_color"], selector=dict(type='candlestick'))
    fig.update_traces(decreasing_fillcolor=theme["down_color"], selector=dict(type='candlestick'))

    fig.update_xaxes(color=theme["xaxes_color"]) 
    fig.update_yaxes(color=theme["yaxes_color"])       
    
    return fig


def _add_nshape_pattern_plot(row: Union[tuple, pd.DataFrame], fig: go.Candlestick) -> go.Candlestick:
    """Add the N-shape pattern elements (A->B->C->D polyline + entry line/marker) to the figure object."""

    if isinstance(row, pd.DataFrame):
        n_idx = row["nshape_idx"].tolist()[0].tolist()
        n_pts = row["nshape_point"].tolist()[0].tolist()
        entry_level = float(row["nshape_entry_level"].tolist()[0]) if "nshape_entry_level" in row.columns and not pd.isna(row["nshape_entry_level"].tolist()[0]) else np.nan
        entry_idx = int(row["nshape_entry_idx"].tolist()[0]) if "nshape_entry_idx" in row.columns and not pd.isna(row["nshape_entry_idx"].tolist()[0]) else -1
    else:
        n_idx = row[1]["nshape_idx"].tolist() if hasattr(row[1]["nshape_idx"], "tolist") else row[1]["nshape_idx"]
        n_pts = row[1]["nshape_point"].tolist() if hasattr(row[1]["nshape_point"], "tolist") else row[1]["nshape_point"]
        entry_level = float(row[1]["nshape_entry_level"]) if "nshape_entry_level" in row[1].index and not pd.isna(row[1]["nshape_entry_level"]) else np.nan
        entry_idx = int(row[1]["nshape_entry_idx"]) if "nshape_entry_idx" in row[1].index and not pd.isna(row[1]["nshape_entry_idx"]) else -1

    if not n_idx or len(n_idx) < 4:
        return fig

    # First 4 points are A,B,C,D. Fifth point (optional) is E retest idx.
    abcd_x = [int(x) for x in n_idx[:4]]
    abcd_y = [float(y) for y in n_pts[:4]]

    fig.add_scatter(
        x=abcd_x,
        y=abcd_y,
        mode="lines",
        name=None,
        line=dict(color="royalblue", width=4),
        showlegend=False,
    )

    if not pd.isna(entry_level):
        # Draw entry level from A to (E if present) else D
        x0 = int(abcd_x[0])
        x1 = int(entry_idx) if entry_idx != -1 else int(abcd_x[-1])
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=float(entry_level),
            y1=float(entry_level),
            line=dict(color="white", width=2),
        )

    if entry_idx != -1 and not pd.isna(entry_level):
        fig.add_scatter(
            x=[int(entry_idx)],
            y=[float(entry_level)],
            mode="markers",
            marker=dict(size=14, color="yellow", symbol="square"),
            name=None,
            showlegend=False,
        )

    return fig



def _plot_candlestick(ohlc: pd.DataFrame, plot_obs:int = 500, fig =None ) -> go.Candlestick:
    """
    :params ohlc is a dataframe with Open, High, Low, Close 
    :type :pd.DataFrame
    
    :params plot_obs is the total number of observations to plot
    :type :int 
    
    :return (go.Candlestick)       
    """
    
    # Check if OHLC columns are present
    check_ohlc_names(ohlc)
    
    # Has the user run find_all_pivot_points?
    if ohlc.columns.str.contains("pivot_pos").sum() == 0:
        print(f"-> Column `pivot_pos` was not found. Did you run `find_all_pivot_points`?")
        sys.exit() 
        
      
    # Find the number of obs. Only plot 500 observations
    if len(ohlc)  > plot_obs + 1:
        print(f"Note only the {plot_obs} points will be plotted")
        
        ohlc = ohlc.iloc[:plot_obs,]
    
    
    # Plot the candlesticks
    if fig is not None :
        fig.add_trace(go.Candlestick(
                    x     = ohlc.index,
                    open  = ohlc.open,
                    high  = ohlc.high,
                    low   = ohlc.low,
                    close = ohlc.close, name="OHLC"    
            ))
    else:
        fig = go.Figure(data=[go.Candlestick(
                    x     = ohlc.index,
                    open  = ohlc.open,
                    high  = ohlc.high,
                    low   = ohlc.low,
                    close = ohlc.close, name="OHLC"    
            )])
    
    
    return fig 


def _add_sbi_pattern_plot(row: Union[tuple, pd.DataFrame], fig: go.Candlestick) -> go.Candlestick:
    """Add the SBI pattern elements (pivot polyline + OB box) to the figure object."""

    if isinstance(row, pd.DataFrame):
        sbi_idx = row["sbi_idx"].tolist()[0].tolist()
        sbi_pts = row["sbi_point"].tolist()[0].tolist()
        ob_idx = int(row["sbi_ob_idx"].tolist()[0])
        ob_high = float(row["sbi_ob_high"].tolist()[0])
        ob_low = float(row["sbi_ob_low"].tolist()[0])
        entry_idx = int(row["sbi_entry_idx"].tolist()[0]) if "sbi_entry_idx" in row.columns and not pd.isna(row["sbi_entry_idx"].tolist()[0]) else -1
        entry_price = float(row["sbi_entry_price"].tolist()[0]) if "sbi_entry_price" in row.columns and not pd.isna(row["sbi_entry_price"].tolist()[0]) else np.nan
        sbi_type = str(row["sbi_type"].tolist()[0]) if "sbi_type" in row.columns else ""
        liq_level = float(row["sbi_liq_level"].tolist()[0]) if "sbi_liq_level" in row.columns and not pd.isna(row["sbi_liq_level"].tolist()[0]) else np.nan
        liq_idx = int(row["sbi_liq_idx"].tolist()[0]) if "sbi_liq_idx" in row.columns and not pd.isna(row["sbi_liq_idx"].tolist()[0]) else -1
        sweep_idx = int(row["sbi_sweep_idx"].tolist()[0]) if "sbi_sweep_idx" in row.columns and not pd.isna(row["sbi_sweep_idx"].tolist()[0]) else -1
        sweep_price = float(row["sbi_sweep_price"].tolist()[0]) if "sbi_sweep_price" in row.columns and not pd.isna(row["sbi_sweep_price"].tolist()[0]) else np.nan
        entry_status = str(row["sbi_entry_status"].tolist()[0]) if "sbi_entry_status" in row.columns else ""
        internal_bos_idx = int(row["sbi_internal_bos_idx"].tolist()[0]) if "sbi_internal_bos_idx" in row.columns and not pd.isna(row["sbi_internal_bos_idx"].tolist()[0]) else -1
        internal_bos_level = float(row["sbi_internal_bos_level"].tolist()[0]) if "sbi_internal_bos_level" in row.columns and not pd.isna(row["sbi_internal_bos_level"].tolist()[0]) else np.nan
    else:
        sbi_idx = row[1]["sbi_idx"].tolist() if hasattr(row[1]["sbi_idx"], "tolist") else row[1]["sbi_idx"]
        sbi_pts = row[1]["sbi_point"].tolist() if hasattr(row[1]["sbi_point"], "tolist") else row[1]["sbi_point"]
        ob_idx = int(row[1]["sbi_ob_idx"]) if not pd.isna(row[1]["sbi_ob_idx"]) else -1
        ob_high = float(row[1]["sbi_ob_high"]) if not pd.isna(row[1]["sbi_ob_high"]) else np.nan
        ob_low = float(row[1]["sbi_ob_low"]) if not pd.isna(row[1]["sbi_ob_low"]) else np.nan
        entry_idx = int(row[1]["sbi_entry_idx"]) if "sbi_entry_idx" in row[1].index and not pd.isna(row[1]["sbi_entry_idx"]) else -1
        entry_price = float(row[1]["sbi_entry_price"]) if "sbi_entry_price" in row[1].index and not pd.isna(row[1]["sbi_entry_price"]) else np.nan
        sbi_type = str(row[1]["sbi_type"]) if "sbi_type" in row[1].index else ""
        liq_level = float(row[1]["sbi_liq_level"]) if "sbi_liq_level" in row[1].index and not pd.isna(row[1]["sbi_liq_level"]) else np.nan
        liq_idx = int(row[1]["sbi_liq_idx"]) if "sbi_liq_idx" in row[1].index and not pd.isna(row[1]["sbi_liq_idx"]) else -1
        sweep_idx = int(row[1]["sbi_sweep_idx"]) if "sbi_sweep_idx" in row[1].index and not pd.isna(row[1]["sbi_sweep_idx"]) else -1
        sweep_price = float(row[1]["sbi_sweep_price"]) if "sbi_sweep_price" in row[1].index and not pd.isna(row[1]["sbi_sweep_price"]) else np.nan
        entry_status = str(row[1]["sbi_entry_status"]) if "sbi_entry_status" in row[1].index else ""
        internal_bos_idx = int(row[1]["sbi_internal_bos_idx"]) if "sbi_internal_bos_idx" in row[1].index and not pd.isna(row[1]["sbi_internal_bos_idx"]) else -1
        internal_bos_level = float(row[1]["sbi_internal_bos_level"]) if "sbi_internal_bos_level" in row[1].index and not pd.isna(row[1]["sbi_internal_bos_level"]) else np.nan

    # Extend the structure line after BOS towards internal liquidity, sweep and entry.
    # This matches the SBI sketch: break (BOS) -> internal liq -> sweep -> return/pending entry.
    ext_x = [int(x) for x in sbi_idx]
    ext_y = [float(y) for y in sbi_pts]
    if liq_idx != -1 and not pd.isna(liq_level):
        if int(liq_idx) not in ext_x:
            ext_x.append(int(liq_idx))
            ext_y.append(float(liq_level))
    if internal_bos_idx != -1 and not pd.isna(internal_bos_level):
        if int(internal_bos_idx) not in ext_x:
            ext_x.append(int(internal_bos_idx))
            ext_y.append(float(internal_bos_level))
    if sweep_idx != -1 and not pd.isna(sweep_price):
        if int(sweep_idx) not in ext_x:
            ext_x.append(int(sweep_idx))
            ext_y.append(float(sweep_price))
    if entry_idx != -1 and not pd.isna(entry_price):
        if int(entry_idx) not in ext_x:
            ext_x.append(int(entry_idx))
            ext_y.append(float(entry_price))

    fig.add_scatter(
        x=ext_x,
        y=ext_y,
        mode="lines",
        name=None,
        line=dict(color="royalblue", width=4),
        showlegend=False,
    )

    if internal_bos_idx != -1 and not pd.isna(internal_bos_level):
        fig.add_scatter(
            x=[int(internal_bos_idx)],
            y=[float(internal_bos_level)],
            mode="markers",
            marker=dict(size=10, color="white", symbol="circle"),
            name=None,
            showlegend=False,
        )

    # Liquidity level (the level that gets swept)
    if liq_idx != -1 and not pd.isna(liq_level):
        x0 = liq_idx
        x1 = entry_idx if entry_idx != -1 else int(sbi_idx[-1])
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=liq_level,
            y1=liq_level,
            line=dict(color="#7CFF7C", width=2, dash="dot"),
        )

    # Sweep marker
    if sweep_idx != -1 and not pd.isna(sweep_price):
        sweep_color = "#FF6B6B" if sbi_type == "buy" else "#7CFF7C"
        fig.add_scatter(
            x=[int(sweep_idx)],
            y=[float(sweep_price)],
            mode="markers",
            marker=dict(size=14, color=sweep_color, symbol="x"),
            name=None,
            showlegend=False,
        )

    if ob_idx != -1 and not (pd.isna(ob_high) or pd.isna(ob_low)):
        # Plot a simple OB rectangle spanning a small window around the OB candle
        x0 = ob_idx - 1
        x1 = ob_idx + 1
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=ob_low,
            y1=ob_high,
            line=dict(color="orange", width=2),
            fillcolor="rgba(255,165,0,0.25)",
        )

    if entry_idx != -1 and not pd.isna(entry_price):
        symbol = "square" if entry_status == "hit" else "square-open"
        fig.add_scatter(
            x=[int(entry_idx)],
            y=[float(entry_price)],
            mode="markers",
            marker=dict(size=14, color="yellow", symbol=symbol),
            name=None,
            showlegend=False,
        )

    return fig


def _plot_pivot_points(ohlc: pd.DataFrame, fig: go.Candlestick, pivot_name: int = "pivot") -> go.Candlestick:
    """
    Plot the pivot points. It are assumes there is a "pivot" column
    
    :params ohlc is a dataframe with Open, High, Low, Close 
    :type :pd.DataFrame
    
    :params fig is the figure object that has the candlestick
    :type :go.Candlestick
    
    :params pivot_name is the name of the column that has the pivot points. Note the column should have int values
            where `1` is pivot lows and `2` is pivot highs
    :type :str 
    
    
    :return (go.Candlestick)
    """
    try:
        pivot_lows  = ohlc.loc[ohlc[pivot_name] == 1,]
        pivot_highs = ohlc.loc[ohlc[pivot_name] == 2,]
    except Exception as e:
        print(f"No column named `{pivot_name}`. Did you run `find_all_pivot_points`?")
        sys.exit() 
        
    fig.add_scatter(
            x = pivot_lows.index ,
            y = pivot_lows[f"{pivot_name}_pos"],
            mode="markers", marker=dict(size=20, color="red"), name="Pivot Low"
        )   
    
    fig.add_scatter(
            x=pivot_highs.index,
            y=pivot_highs[f"{pivot_name}_pos"] ,
            mode="markers", marker=dict(size=20, color="green"), name="Pivot High"
        )    

    return fig 

def display_pivot_points(ohlc : pd.DataFrame, 
                         theme: Dict[str, str] = {"bg_color": "black", "up_color":"#3D9970", 
                                                      "down_color": "#FF4136", "legend_font_color": "white", "xaxes_color": "white", "yaxes_color": "white"}, 
                         plot_obs:int = 500 ) -> None:
    """
    Display the pivot points and the OHLC data in a graph
    
    :params ohlc is a dataframe with Open, High, Low, Close and the pivot_pos
    :type :pd.DataFrame
    
    :params theme is the set of parameters to set the aesthetics of the graph
    :type :Dict[str, str]
    
    :params plot_obs is the total number of observations to plot
    :type :int 
    
    :return (None)
    """
    
    # Get the Candlestick figure object
    fig = _plot_candlestick(ohlc, plot_obs)
    
    # Add the pivot points 
    fig = _plot_pivot_points(ohlc, fig)    
                
    # Set the theme   
    fig = set_theme(fig, theme)
    
    
    fig.show()
    

def _add_head_shoulder_pattern_plot(row: Union[tuple, pd.DataFrame], fig: go.Candlestick, 
                                    x_val: str = "hs_idx", y_val: str = "hs_point") -> go.Candlestick:
    """
    Add the Head and Shoulders pattern to the given figure object
    
    :params row is either a pandas dataframe or a row that has the flag chart pattern info.
    :type :Union[tuple, pd.DataFrame]
    
    :params fig is the figure object
    :type :go.Candlestick
    
    :return (go.Candlestick)   
    """
    
    if isinstance(row, tuple):
        x_values = row[1][x_val]
        y_values = row[1][y_val]
    else:
        x_values = row[x_val]
        y_values = row[y_val]       
    
    fig.add_scatter(
            x = x_values , y = y_values,
                    mode='lines',
                    name=None, line=dict(color='royalblue', width=4), showlegend=False )   
        
    return fig   

def _add_doubles_pattern_plot(row: Union[tuple, pd.DataFrame], fig: go.Candlestick) -> go.Candlestick:
    """
    Add the the Double chart patterns to the given figure object
    
    :params row is either a pandas dataframe or a row that has the flag chart pattern info.
    :type :Union[tuple, pd.DataFrame]
    
    :params fig is the figure object
    :type :go.Candlestick
    
    :return (go.Candlestick)    
    """
    
    if isinstance(row, tuple):
        double_idx = row[1]["double_idx"]
        double_pts = row[1]["double_point"]
    else:
        double_idx = row["double_idx"]
        double_pts = row["double_point"] 
    
    fig.add_scatter(x = double_idx , y = double_pts,
                    mode='lines',
                    name=None, line=dict(color='royalblue', width=4), showlegend=False )     
            
    return fig 

def _add_triangle_pattern_plot(row: Union[tuple, pd.DataFrame], fig: go.Candlestick) -> go.Candlestick:
    """
    Add the triangle pattern to the figure object 
    
    :params row is either a pandas dataframe or a row that has the flag chart pattern info.
    :type :Union[tuple, pd.DataFrame]
    
    :params fig is the figure object
    :type :go.Candlestick
    
    :return (go.Candlestick)    
    """
    
    if isinstance(row, pd.DataFrame):
        high_idx  = row["triangle_high_idx"]
        low_idx   = row["triangle_low_idx"]
        intercmin = row["triangle_intercmin"]
        intercmax = row["triangle_intercmax"]
        slmax     = row["triangle_slmax"]
        slmin     = row["triangle_slmin"]
        
    else:
        high_idx  = row[1]["triangle_high_idx"].tolist()
        low_idx   = row[1]["triangle_low_idx"].tolist()
        intercmin = row[1]["triangle_intercmin"]
        intercmax = row[1]["triangle_intercmax"]
        slmax     = row[1]["triangle_slmax"]
        slmin     = row[1]["triangle_slmin"]

    fig.add_scatter(x = [int(idx) for idx in high_idx] , y = [int(idx)*slmax + intercmax for idx in high_idx],
                    mode='lines',
                    name=None, line=dict(color='royalblue', width=4), showlegend=False )
    
    fig.add_scatter(x = [int(idx) for idx in low_idx] , y = [int(idx)*slmin + intercmin for idx in low_idx],
                    mode='lines',
                    name=None, line=dict(color='royalblue', width=4), showlegend=False )   
    
    return fig 


def _add_pennant_pattern_plot(row: Union[tuple, pd.DataFrame], fig: go.Candlestick) -> go.Candlestick:
    """
     Add the pennant pattern to the figure object
    
    :params row is either a pandas dataframe or a row that has the pennant chart pattern info.
    :type :Union[tuple, pd.DataFrame]
    
    :params fig is the figure object
    :type :go.Candlestick
    
    :return (go.Candlestick)   
    """
    
    if isinstance(row, pd.DataFrame):
    
        x_low_vals      = row["pennant_lows_idx"].tolist()[0].tolist()
        y_low_vals_arr  = row["pennant_slmin"]*row["pennant_lows_idx"] + row["pennant_intercmin"]
        y_low_vals      = y_low_vals_arr.tolist()[0].tolist()
        
        x_high_vals      = row["pennant_highs_idx"].tolist()[0].tolist()
        y_high_vals_arr  = row["pennant_slmax"]*row["pennant_highs_idx"] + row["pennant_intercmax"]
        y_high_vals      = y_high_vals_arr.tolist()[0].tolist()    
        
    else:
        
        x_low_vals      = row[1]["pennant_lows_idx"]
        y_low_vals_arr  = row[1]["pennant_slmin"]*row[1]["pennant_lows_idx"] + row[1]["pennant_intercmin"]
        y_low_vals      = y_low_vals_arr
        
        x_high_vals      = row[1]["pennant_highs_idx"]
        y_high_vals_arr  = row[1]["pennant_slmax"]*row[1]["pennant_highs_idx"] + row[1]["pennant_intercmax"]
        y_high_vals      = y_high_vals_arr   
        
    fig.add_scatter(x = x_low_vals , y = y_low_vals,
                    mode='lines',
                    name=None, line=dict(color='royalblue', width=4), showlegend=False )
    
    fig.add_scatter(x = x_high_vals , y = y_high_vals,
                    mode='lines',
                    name=None, line=dict(color='royalblue', width=4), showlegend=False)      

    return fig 


def _add_flag_pattern_plot(row: Union[tuple, pd.DataFrame], fig: go.Candlestick) -> go.Candlestick:
    """
    Add the flag pattern to the figure object
    
    :params row is either a pandas dataframe or a row that has the flag chart pattern info.
    :type :Union[tuple, pd.DataFrame]
    
    :params fig is the figure object
    :type :go.Candlestick
    
    :return (go.Candlestick)
    """
    
    if isinstance(row, pd.DataFrame):

        x_low_vals      = row["flag_lows_idx"].tolist()[0].tolist()
        y_low_vals_arr  = row["flag_slmin"]*row["flag_lows_idx"] + row["flag_intercmin"]
        y_low_vals      = y_low_vals_arr.tolist()[0].tolist()
        
        x_high_vals      = row["flag_highs_idx"].tolist()[0].tolist()
        y_high_vals_arr  = row["flag_slmax"]*row["flag_highs_idx"] + row["flag_intercmax"]
        y_high_vals      = y_high_vals_arr.tolist()[0].tolist()    
        
    else:
        
        x_low_vals      = row[1]["flag_lows_idx"]
        y_low_vals_arr  = row[1]["flag_slmin"]*row[1]["flag_lows_idx"] + row[1]["flag_intercmin"]
        y_low_vals      = y_low_vals_arr
        
        x_high_vals      = row[1]["flag_highs_idx"]
        y_high_vals_arr  = row[1]["flag_slmax"]*row[1]["flag_highs_idx"] + row[1]["flag_intercmax"]
        y_high_vals      = y_high_vals_arr   
        
    fig.add_scatter(x = x_low_vals , y = y_low_vals,
                    mode='lines',
                    name=None, line=dict(color='royalblue', width=4), showlegend=False )
    
    fig.add_scatter(x = x_high_vals , y = y_high_vals,
                    mode='lines',
                    name=None, line=dict(color='royalblue', width=4), showlegend=False)      

    return fig 

def save_chart_pattern(fig: go.Candlestick, pattern: str, row: Union[None,tuple], image_subdir: Union[None, str] = None) -> None:
    """
    Save the chart pattern plot
    
    :params fig is the Candlestick object
    :type :go.Candlestick
    
    :params pattern is the name of the chart pattern to save
    :type :str
    
    :params row is the pandas Series that has the index value
    :type :Union[None,tuple]
    
    :return (None)
    """
    
    out_dir_name = image_subdir if image_subdir else pattern

    # Create the images/flag folder if it does not exist
    if not os.path.exists(os.path.join(os.path.realpath(''), "images")):
            os.mkdir(os.path.join(os.path.realpath(''), "images"))
           
    if not  os.path.exists(os.path.join(os.path.realpath(''), "images", out_dir_name)):
         os.mkdir(os.path.join(os.path.realpath(''), "images", out_dir_name))
    
    if row:        
        fig.write_image(os.path.join(os.path.realpath(''), "images", out_dir_name, f"fig{row[0]}.png"))
    else:
        fig.write_image(os.path.join(os.path.realpath(''), "images", out_dir_name, f"fig-{out_dir_name}.png"))
                   
def display_chart_pattern(ohlc: pd.DataFrame, pattern: str = "flag", 
                          save: bool = True, lookback: int = 60, pivot_name: str = "pivot", image_subdir: Union[None, str] = None) -> None:
    """
    Display the specified chart pattern. 
    
    :params ohlc is the dataframe that contains the OHLC data and the chart pattern points
    :type :pd.DataFrame
    
    :params pattern is the name of the pattern to plot 
    :type :str 
    
    :params save is whether to save the plot(s) and not display them
    :type :bool
    
    :params lookback is the number of candlesticks to plot
    :type :int
    
    :params pivot_name is the name of the column that has the pivot points. Note the column should have int values
            where `1` is pivot lows and `2` is pivot highs
    :type :str 
    
    
    :return (None)
    """
    
    # Check if the columns have the `pattern` results
    if ohlc.columns.str.lower().str.contains(pattern).sum() == 0:
        print(f"No columns for the pattern `{pattern}`. Did you run the function to get the pattern?")
        sys.exit()
    

    if pattern == "flag":
        pattern_points = ohlc.loc[ohlc["chart_type"]== "flag"]
    elif pattern == "double":
        pattern_points = ohlc.loc[ohlc["chart_type"]== "double"]
    elif pattern == "hs":
        pattern_points = ohlc.loc[ohlc["chart_type"]=="hs"]
    elif pattern == "ihs":
        pattern_points = ohlc.loc[ohlc["chart_type"]=="ihs"]        
    elif pattern == "triangle":
        pattern_points = ohlc.loc[ohlc["chart_type"]=="triangle"]
    elif pattern == "pennant":
        pattern_points = ohlc.loc[ohlc["chart_type"]=="pennant"]
    elif pattern == "sbi":
        pattern_points = ohlc.loc[ohlc["chart_type"]=="sbi"]
    elif pattern == "nshape":
        pattern_points = ohlc.loc[ohlc["chart_type"]=="nshape"]
            
    
    if len(pattern_points) == 0: # There is no pattern found
        print(f"There are no `{pattern}` patterns detected.")
    elif len(pattern_points) == 1:
        # Get the Candlestick figure object
        fig = _plot_candlestick(ohlc)
        
        # Add the pivot points 
        fig = _plot_pivot_points(ohlc, fig, pivot_name)    
                    
        # Set the theme   
        fig = set_theme(fig)
            
        if pattern == "flag":
            # Plot the Flag pattern
            fig = _add_flag_pattern_plot(pattern_points, fig)
        elif pattern == "double":
            fig = _add_doubles_pattern_plot(pattern_points, fig)
        elif pattern == "hs":
            fig = _add_head_shoulder_pattern_plot(pattern_points, fig)
        elif pattern == "ihs":
            fig  = _add_head_shoulder_pattern_plot(pattern_points, fig, "ihs_idx", "ihs_point")
        elif pattern == "triangle":
                fig  = _add_triangle_pattern_plot(pattern_points, fig)
        elif pattern == "pennant":
            fig = _add_pennant_pattern_plot(pattern_points, fig)
        elif pattern == "sbi":
            fig = _add_sbi_pattern_plot(pattern_points, fig)
        elif pattern == "nshape":
            fig = _add_nshape_pattern_plot(pattern_points, fig)
                
        if save:
            save_chart_pattern(fig, pattern, None, image_subdir=image_subdir)
        else:
            fig.show()
    elif len(pattern_points) > 1:
             
      for row in tqdm(pattern_points.iterrows(), desc=f"Saving the {pattern} charts..."):
            # Get the row index
            pattern_point = row[0]
            
            # Make sure at least 50 candlesticks are available, if possible
            if pattern_point - lookback < 0:
                start  = 0
            else:
                start = pattern_point - lookback
            
            # Get a subset of the ohlc plus chart pattens included
            ohlc_copy = ohlc.loc[start:pattern_point,]
            
            # Add the ohlc data
            fig = _plot_candlestick(ohlc_copy)
                
            # Add the pivot points 
            fig = _plot_pivot_points(ohlc_copy, fig, pivot_name)    
                        
            # Set the theme   
            fig = set_theme(fig)   
            
            if pattern == "flag":    
                # Add the flag pattern 
                fig = _add_flag_pattern_plot(row, fig)
            elif pattern == "double":
                # Add the double pattern         
                fig = _add_doubles_pattern_plot(row, fig)
            elif pattern == "hs":
                fig = _add_head_shoulder_pattern_plot(row, fig)
            elif pattern == "ihs":
                fig  = _add_head_shoulder_pattern_plot(row, fig, "ihs_idx", "ihs_point")
            elif pattern == "triangle":
                fig  = _add_triangle_pattern_plot(row, fig)
            elif pattern == "pennant":
                fig = _add_pennant_pattern_plot(row, fig)
            elif pattern == "sbi":
                fig = _add_sbi_pattern_plot(row, fig)
            elif pattern == "nshape":
                fig = _add_nshape_pattern_plot(row, fig)
            
            # Save the figures 
            save_chart_pattern(fig, pattern, row, image_subdir=image_subdir)
                
      if save:
        return
              
