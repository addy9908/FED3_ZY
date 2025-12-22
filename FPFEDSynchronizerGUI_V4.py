'''
V3: with peak function
V4: add message log panel

Author: Zengyou Ye
Contact: addy9908@gmail.com

'''

import os
import traceback
# import csv
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, END, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal #, stats
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import zscore #, linregress
from pyexcelerate import Workbook
# import itertools
# from collections import defaultdict

# --------- Utilities ---------
def safe_read_csv(filepath, expected_min_cols=3):
    """
    Reads a CSV with automatic delimiter detection and vector-optimized 
    timestamp correction.
    """
    try:
        # 'sep=None' with 'engine=python' enables auto-detection of delimiters
        df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='warn')
        
        # Fallback if auto-detection fails to find enough columns
        if df.shape[1] < expected_min_cols:
            df = pd.read_csv(filepath) 
            
    except Exception as e:
        print(f"Read error: {e}")
        return pd.DataFrame()

    # 1. Efficient Cleaning
    # Drop rows where the first column is NaN and drop fully empty columns
    df = df.dropna(subset=[df.columns[0]]).dropna(axis=1, how='all')
    
    # 2. Optimized Timestamp Correction
    comp_col = 'ComputerTimestamp'
    sys_col = 'SystemTimestamp'
    
    if comp_col in df.columns and sys_col in df.columns:
        # Detect where the clock resets (diff < 0)
        is_reset = df[comp_col].diff().fillna(0) < 0
        
        if is_reset.any():
            print('Detected timestamp resets. Applying vectorized fix...')
            
            # Calculate the gap (delta) at every point where a reset occurs
            # Formula: (Prev_Comp + (Curr_Sys - Prev_Sys)*1000) - Curr_Comp
            # We use .shift(1) to get "previous" values without looping
            prev_comp = df[comp_col].shift(1)
            prev_sys = df[sys_col].shift(1)
            
            deltas = (prev_comp + (df[sys_col] - prev_sys) * 1000) - df[comp_col]
            
            # Only keep deltas at reset points, others are 0
            reset_deltas = np.where(is_reset, deltas, 0)
            
            # Use cumsum to propagate the correction to all subsequent rows
            # This handles multiple resets in one file efficiently
            total_correction = np.cumsum(reset_deltas)
            df[comp_col] = df[comp_col] + total_correction

    return df.reset_index(drop=True)

def safe_read_csv_old(filepath, expected_min_cols=3):
    try:
        df = pd.read_csv(filepath, sep=';') # used in France as delimiter of csv
        if df.shape[1] < expected_min_cols:
            raise ValueError("Too few columns after parsing with ; — trying default separator.")
    except Exception as e:
        print(f"Fallback triggered: {e}")
        df = pd.read_csv(filepath)  # default separator
    
    # Clean: drop rows and columns that are all NaN
    df = df.dropna(subset=[df.columns[0]])
    df = df.dropna(axis=1, how='all')  # drop columns where all values are NaN
    
    # for some reason (could be a limit of max value), the computer time in FP mess up
    # we need to fix here
    comp_col = 'ComputerTimestamp'
    sys_col = 'SystemTimestamp'
    if comp_col in df.columns:
        # Find first index where time is not sorted
        not_sorted_idx = df.index[df[comp_col].diff().fillna(0) < 0]
        
        if not not_sorted_idx.empty:
            print('Found not sorted computerTime. Fix now...')
            not_sorted_idx = list(not_sorted_idx) + [len(df)]
        
            for i in range(len(not_sorted_idx) - 1):
                start = not_sorted_idx[i]
                end = not_sorted_idx[i + 1]  # exclusive
        
                # Skip if start is 0 (no previous row to reference)
                if start == 0:
                    continue
        
                delta = (
                    df.at[start - 1, comp_col]
                    + df.at[start, sys_col] * 1000
                    - df.at[start - 1, sys_col]*1000
                    - df.at[start, comp_col]
                )
        
                df.loc[start:end, comp_col] += delta
            
    df = df.reset_index(drop=True)
    return df    

def load_merge_sort(file_list):
    if not file_list:
        return pd.DataFrame()
    
    df = pd.concat([safe_read_csv(f) for f in file_list], ignore_index=True)
    return df.sort_values(df.columns[1]).reset_index(drop=True)

def add_files_to_listbox(listbox, files):
    listbox.delete(0, END)
    for f in files:
        listbox.insert(END, str(f.name))
        
def extract_event_epochs_from_columns(ts_data, event_columns, time_col='time', 
                                      value_col='value', pre_time=4, post_time=10,
                                      limit_to_available=False, zscore_method='none',
                                      zscore_col=None):
    """
    Extract epochs around events from time series data where events are stored as columns.
    
    Parameters:
    -----------
    ts_data : pd.DataFrame
        Time series data with time, value, and event columns (DIO_xxx)
    event_columns : list of str
        List of column names containing event markers (e.g., ['DIO_001', 'DIO_002'])
    time_col : str
        Name of time column
    value_col : str
        Name of value column
    pre_time : float
        Time before event (seconds)
    post_time : float
        Time after event (seconds)
    limit_to_available : bool
        If True, limit epoch to actual available data before/after each event
        If False, use fixed pre_time and post_time for all epochs
    zscore_method : str
        'none': Use raw signal
        'full': Use pre-calculated z-score from entire trace (requires zscore_col)
        'baseline': Calculate z-score using pre-event window as baseline
    zscore_col : str or None
        Column name containing pre-calculated z-scores (required if zscore_method='full')
    
    Returns:
    --------
    dict with event column names as keys, each containing dict with:
        'onset' and/or 'offset' subdicts, each containing:
            - DataFrame with time as index and trials as columns (t1, t2, t3, ...)
    """
    results = {}
    
    # Get sampling rate and time array
    time_array = ts_data[time_col].values
    dt = np.median(np.diff(time_array))
    
    # Determine which signal to use
    if zscore_method == 'full':
        if zscore_col is None or zscore_col not in ts_data.columns:
            raise ValueError(f"zscore_method='full' requires valid zscore_col. Got: {zscore_col}")
        signal = ts_data[zscore_col].values
    else:
        signal = ts_data[value_col].values
    
    # Calculate number of samples for fixed window
    n_pre = int(pre_time / dt)
    n_post = int(post_time / dt)
    
    # Create common time axis for all epochs
    common_time_axis = np.arange(-n_pre, n_post) * dt
    
    for event_col in event_columns:
        if event_col not in ts_data.columns:
            print(f"Warning: {event_col} not found in dataframe")
            continue
        
        event_data = ts_data[event_col].values
        results[event_col] = {}
        
        # Find onset events (transitions from 0 to 1)
        onset_indices = np.where(np.diff(event_data) == 1)[0] + 1
        
        if len(onset_indices) > 0:
            onset_epochs = []
            
            for idx in onset_indices:
                # Initialize epoch with NaN
                epoch = np.full(len(common_time_axis), np.nan)
                
                if limit_to_available:
                    # Find previous offset (or start of recording)
                    prev_offset_idx = 0
                    for i in range(idx - 1, -1, -1):
                        if event_data[i] == 0 and (i == 0 or event_data[i-1] == 1):
                            prev_offset_idx = i
                            break
                    
                    # Find next onset (or end of recording)
                    next_onset_idx = len(event_data) - 1
                    for i in range(idx + 1, len(event_data)):
                        if event_data[i] == 1 and event_data[i-1] == 0:
                            next_onset_idx = i - 1
                            break
                    
                    # Calculate available pre and post samples
                    avail_pre = min(idx - prev_offset_idx, n_pre)
                    avail_post = min(next_onset_idx - idx + 1, n_post)
                    
                    start_idx = idx - avail_pre
                    end_idx = idx + avail_post
                    
                    extracted_epoch = signal[start_idx:end_idx]
                    
                    # Apply baseline z-score if requested
                    if zscore_method == 'baseline' and avail_pre > 0:
                        baseline = extracted_epoch[:avail_pre]
                        if len(baseline) > 1:
                            baseline_mean = np.mean(baseline)
                            baseline_std = np.std(baseline)
                            if baseline_std > 0:
                                extracted_epoch = (extracted_epoch - baseline_mean) / baseline_std
                    
                    # Place extracted epoch in the correct position of the full epoch array
                    # Fill available pre-event data
                    epoch[n_pre - avail_pre:n_pre + avail_post] = extracted_epoch
                else:
                    # Fixed window
                    if idx >= n_pre and idx + n_post <= len(signal):
                        extracted_epoch = signal[idx - n_pre:idx + n_post]
                        
                        # Apply baseline z-score if requested
                        if zscore_method == 'baseline':
                            baseline = extracted_epoch[:n_pre]
                            if len(baseline) > 1:
                                baseline_mean = np.mean(baseline)
                                baseline_std = np.std(baseline)
                                if baseline_std > 0:
                                    extracted_epoch = (extracted_epoch - baseline_mean) / baseline_std
                        
                        epoch = extracted_epoch
                    else:
                        # Skip this event if not enough data
                        continue
                
                onset_epochs.append(epoch)
            
            if onset_epochs:
                # Create DataFrame with time as index and trials as columns
                onset_df = pd.DataFrame(
                    np.array(onset_epochs).T,
                    index=common_time_axis,
                    columns=[f't{i+1}' for i in range(len(onset_epochs))]
                )
                onset_df.index.name = time_col
                results[event_col]['onset'] = onset_df
        
        # Find offset events (transitions from 1 to 0)
        offset_indices = np.where(np.diff(event_data) == -1)[0] + 1
        
        if len(offset_indices) > 0:
            offset_epochs = []
            
            for idx in offset_indices:
                # Initialize epoch with NaN
                epoch = np.full(len(common_time_axis), np.nan)
                
                if limit_to_available:
                    # Find previous onset
                    prev_onset_idx = 0
                    for i in range(idx - 1, -1, -1):
                        if event_data[i] == 1 and (i == 0 or event_data[i-1] == 0):
                            prev_onset_idx = i
                            break
                    
                    # Find next offset
                    next_offset_idx = len(event_data) - 1
                    for i in range(idx + 1, len(event_data)):
                        if event_data[i] == 0 and event_data[i-1] == 1:
                            next_offset_idx = i - 1
                            break
                    
                    avail_pre = min(idx - prev_onset_idx, n_pre)
                    avail_post = min(next_offset_idx - idx + 1, n_post)
                    
                    start_idx = idx - avail_pre
                    end_idx = idx + avail_post
                    
                    extracted_epoch = signal[start_idx:end_idx]
                    
                    # Apply baseline z-score if requested
                    if zscore_method == 'baseline' and avail_pre > 0:
                        baseline = extracted_epoch[:avail_pre]
                        if len(baseline) > 1:
                            baseline_mean = np.mean(baseline)
                            baseline_std = np.std(baseline)
                            if baseline_std > 0:
                                extracted_epoch = (extracted_epoch - baseline_mean) / baseline_std
                    
                    # Place extracted epoch in the correct position
                    epoch[n_pre - avail_pre:n_pre + avail_post] = extracted_epoch
                else:
                    if idx >= n_pre and idx + n_post <= len(signal):
                        extracted_epoch = signal[idx - n_pre:idx + n_post]
                        
                        # Apply baseline z-score if requested
                        if zscore_method == 'baseline':
                            baseline = extracted_epoch[:n_pre]
                            if len(baseline) > 1:
                                baseline_mean = np.mean(baseline)
                                baseline_std = np.std(baseline)
                                if baseline_std > 0:
                                    extracted_epoch = (extracted_epoch - baseline_mean) / baseline_std
                        
                        epoch = extracted_epoch
                    else:
                        continue
                
                offset_epochs.append(epoch)
            
            if offset_epochs:
                # Create DataFrame with time as index and trials as columns
                offset_df = pd.DataFrame(
                    np.array(offset_epochs).T,
                    index=common_time_axis,
                    columns=[f't{i+1}' for i in range(len(offset_epochs))]
                )
                offset_df.index.name = time_col
                results[event_col]['offset'] = offset_df
    
    return results

def plot_event_triggered_average(event_dict, title='Event-Triggered Average', 
                                 ylabel='Signal', figsize=None):
    """
    Plot heatmap and mean±SEM for event-triggered data.
    
    Parameters:
    -----------
    event_dict : dict
        Dictionary with 'onset' and/or 'offset' keys, each containing DataFrame
    title : str
        Plot title
    ylabel : str
        Y-axis label for signal
    """
    n_plots = len(event_dict)
    if n_plots == 0:
        print("No data to plot")
        return None, None
    
    if figsize is None:
        figsize = (12, 4 * n_plots)
    
    fig, axes = plt.subplots(n_plots, 2, figsize=figsize)
    
    # Handle single plot case
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (event_type, df) in enumerate(event_dict.items()):
        # Get data from DataFrame
        time_axis = df.index.values
        epochs_matrix = df.values  # time x trials
        n_trials = df.shape[1]
        
        # Transpose for plotting (trials x time)
        epochs_for_plot = epochs_matrix.T
        
        # Plot heatmap using imagesc-like approach with AlphaData
        ax_heat = axes[idx, 0]
        
        if n_trials == 1:
            # For single trial
            extent = [time_axis[0], time_axis[-1], 0, 1]
            # Create masked array for NaN values
            masked_data = np.ma.masked_invalid(epochs_for_plot)
            im = ax_heat.imshow(masked_data, aspect='auto', cmap='viridis',
                               extent=extent, origin='lower', interpolation='nearest')
            ax_heat.set_yticks([0.5])
            ax_heat.set_yticklabels(['1'])
        else:
            # Multiple trials - use imagesc with AlphaData for NaN handling
            # Similar to MATLAB: imagesc(xvar, yvar, cvar, 'AlphaData', ~isnan(cvar))
            yvar = np.arange(n_trials)
            xvar = time_axis
            cvar = epochs_for_plot
            
            # Create image with NaN transparency
            im = ax_heat.imshow(cvar, aspect='auto', cmap='viridis',
                               extent=[xvar[0], xvar[-1], yvar[-1] + 0.5, yvar[0] - 0.5],
                               interpolation='nearest')
            
            # Set alpha channel based on NaN values (transparent where NaN)
            alpha_data = ~np.isnan(cvar)
            im.set_array(np.ma.masked_invalid(cvar))
            
            # Set background to white for NaN regions
            ax_heat.set_facecolor('white')
            
            # Set y-tick labels to start from 1
            yticks = ax_heat.get_yticks()
            valid_yticks = yticks[(yticks >= 0) & (yticks < n_trials)]
            ax_heat.set_yticks(valid_yticks)
            ax_heat.set_yticklabels([int(y + 1) for y in valid_yticks])
        
        ax_heat.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_heat.set_xlabel('Time (s)')
        ax_heat.set_ylabel('Trial #')
        ax_heat.set_title(f'{title} - {event_type.capitalize()} (n={n_trials})')
        plt.colorbar(im, ax=ax_heat)#, label=ylabel)
        
        # Plot mean with SEM
        ax_mean = axes[idx, 1]
        # Calculate mean and SEM ignoring NaN values (across trials)
        mean_signal = np.nanmean(epochs_matrix, axis=1)  # axis=1 because epochs_matrix is time x trials
        sem_signal = np.nanstd(epochs_matrix, axis=1) / np.sqrt(np.sum(~np.isnan(epochs_matrix), axis=1))
        
        ax_mean.plot(time_axis, mean_signal, 'b-', linewidth=2, label='Mean')
        ax_mean.fill_between(time_axis,
                            mean_signal - sem_signal,
                            mean_signal + sem_signal,
                            alpha=0.3, color='b', label='±SEM')
        ax_mean.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_mean.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax_mean.set_xlabel('Time (s)')
        ax_mean.set_ylabel(ylabel)
        ax_mean.set_title(f'Average Response - {event_type.capitalize()}')
        # ax_mean.legend()
        ax_mean.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

def plot_signal_with_events(ts_data, signal_col, event_cols, time_col='time',
                           time_range=None, downsample=1, figsize=(14, 8)):
    """
    Plot signal trace with event markers as colored bands.
    
    Parameters:
    -----------
    ts_data : pd.DataFrame
        Time series data with time, signal, and event columns
    signal_col : str
        Name of signal column to plot
    event_cols : list of str
        List of event column names to display
    time_col : str
        Name of time column
    time_range : tuple or None
        (start_time, end_time) to plot, None for entire trace
    downsample : int
        Downsampling factor (1 = no downsampling)
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    from scipy import stats as scipy_stats
    
    # Filter time range
    if time_range is not None:
        mask = (ts_data[time_col] >= time_range[0]) & (ts_data[time_col] <= time_range[1])
        data = ts_data[mask].copy()
    else:
        data = ts_data.copy()
    
    # Downsample if requested
    if downsample > 1:
        data = data.iloc[::downsample].reset_index(drop=True)
    
    time = data[time_col].values
    signal = data[signal_col].values
    
    # Calculate z-score
    signal_zscore = scipy_stats.zscore(signal)
    
    # Create subplots: one for each event + one for signal
    n_events = len(event_cols)
    fig, axes = plt.subplots(n_events + 1, 1, figsize=figsize, 
                             sharex=True, height_ratios=[0.5]*n_events + [2])
    
    if n_events == 0:
        axes = [axes]
    
    # Color palette for different events
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_events, 1)))
    
    # Plot each event type
    for idx, event_col in enumerate(event_cols):
        ax = axes[idx]
        event_data = data[event_col].values
        
        # Find onset-offset pairs
        onsets = np.where(np.diff(event_data) == 1)[0] + 1
        offsets = np.where(np.diff(event_data) == -1)[0] + 1
        
        # Handle edge cases
        if len(event_data) > 0 and event_data[0] == 1:
            onsets = np.concatenate([[0], onsets])
        if len(event_data) > 0 and event_data[-1] == 1:
            offsets = np.concatenate([offsets, [len(event_data) - 1]])
        
        # Plot colored bands for each onset-offset pair
        for onset_idx, offset_idx in zip(onsets, offsets):
            if onset_idx < len(time) and offset_idx < len(time):
                ax.axvspan(time[onset_idx], time[offset_idx], 
                          alpha=0.3, color=colors[idx], linewidth=0)
        
        # Mark onsets with vertical lines
        for onset_idx in onsets:
            if onset_idx < len(time):
                ax.axvline(time[onset_idx], color=colors[idx], 
                          linestyle='-', linewidth=1.5, alpha=0.8)
        
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([])
        ax.set_ylabel(event_col, rotation=0, ha='right', va='center', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
    
    # Plot z-scored signal
    ax_signal = axes[-1]
    ax_signal.plot(time, signal_zscore, 'b-', linewidth=0.8, alpha=0.7)
    ax_signal.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_signal.set_xlabel('Time (s)', fontsize=12)
    ax_signal.set_ylabel('Z-score', fontsize=12)
    ax_signal.set_title(f'{signal_col} - Signal Trace with Events', fontsize=14, pad=10)
    ax_signal.grid(True, alpha=0.3)
    ax_signal.spines['top'].set_visible(False)
    ax_signal.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig, axes

def analyze_signals_by_events(ts_data, signal_cols, time_col='time', 
                             pre_time=4, post_time=10, 
                             limit_to_available=False, ylabel='Signal',
                             plot_traces=True, time_range=None, downsample=1,
                             zscore_method='none'):
    """
    Analyze multiple signal columns with their corresponding event columns.
    
    Parameters:
    -----------
    ts_data : pd.DataFrame
        Time series data with time, signal columns, and event columns
    signal_cols : list of str
        List of signal column names to analyze (e.g., ['G0', 'G1'])
    time_col : str
        Name of time column
    pre_time : float
        Time before event (seconds)
    post_time : float
        Time after event (seconds)
    limit_to_available : bool
        Whether to limit epochs to available data between events
    ylabel : str
        Y-axis label for plots
    plot_traces : bool
        Whether to plot full signal traces with event markers
    time_range : tuple or None
        (start_time, end_time) for trace plots, None for entire trace
    downsample : int
        Downsampling factor for trace plots (1 = no downsampling)
    zscore_method : str
        'none': Use raw signal
        'full': Use pre-calculated z-score from entire trace
        'baseline': Calculate z-score using pre-event window as baseline
    
    Returns:
    --------
    tuple : (all_results, all_figures)
        all_results: dict with structure {signal_col: {'zscore': array, 'events': {event_col: {event_type: epoch_data}}}}
        all_figures: dict with structure {signal_col: {'trace': fig, event_col: fig}}
    """
    from scipy import stats as scipy_stats
    
    all_results = {}
    all_figures = {}
    
    for value_col in signal_cols:
        if value_col not in ts_data.columns:
            print(f"Warning: Signal column '{value_col}' not found in dataframe")
            continue
        
        # Determine event columns based on signal column
        if value_col == 'G0':
            candidate_events = ['DIO_wheel_L', 'DIO_fed_L']
        elif value_col == 'G1':
            candidate_events = ['DIO_wheel_R', 'DIO_fed_R']
        else:
            print(f"Warning: No event mapping defined for '{value_col}'")
            continue
        
        # Check which event columns actually exist in the dataframe
        event_columns = [col for col in candidate_events if col in ts_data.columns]
        
        if len(event_columns) == 0:
            print(f"Warning: No event columns found for '{value_col}' (looked for: {candidate_events})")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing signal: {value_col}")
        print(f"Event columns: {event_columns}")
        print(f"{'='*60}")
        
        # Calculate and store z-score for the entire trace
        signal_zscore = scipy_stats.zscore(ts_data[value_col].values)
        
        # Create zscore DataFrame with time column
        zscore_df = pd.DataFrame({
            time_col: ts_data[time_col].values,
            'zscore': signal_zscore
        })
        zscore_df = zscore_df.set_index(time_col)
        
        # Initialize results for this signal
        all_results[value_col] = {
            'zscore': zscore_df,
            'events': {}
        }
        
        # Temporarily add zscore column to dataframe if using 'full' method
        zscore_col_name = f'{value_col}_zscore'
        if zscore_method == 'full':
            ts_data_with_zscore = ts_data.copy()
            ts_data_with_zscore[zscore_col_name] = signal_zscore
        else:
            ts_data_with_zscore = ts_data
        
        # Initialize figure dict for this signal
        if value_col not in all_figures:
            all_figures[value_col] = {}
        
        # Plot full trace with event markers
        if plot_traces:
            trace_fig, trace_axes = plot_signal_with_events(
                ts_data,
                signal_col=value_col,
                event_cols=event_columns,
                time_col=time_col,
                time_range=time_range,
                downsample=downsample
            )
            all_figures[value_col]['trace'] = trace_fig
        
        # Extract epochs
        results = extract_event_epochs_from_columns(
            ts_data_with_zscore,
            event_columns=event_columns,
            time_col=time_col,
            value_col=value_col,
            pre_time=pre_time,
            post_time=post_time,
            limit_to_available=limit_to_available,
            zscore_method=zscore_method,
            zscore_col=zscore_col_name if zscore_method == 'full' else None
        )
        
        # Store results
        all_results[value_col]['events'] = results
        
        # Plot each event type and store figures
        for event_name, event_dict in results.items():
            if len(event_dict) > 0:  # Check if there are any events
                fig, axes = plot_event_triggered_average(
                    event_dict,
                    title=f'{value_col} - {event_name}',
                    ylabel=ylabel
                )
                all_figures[value_col][event_name] = fig
            else:
                print(f"No valid epochs found for {value_col} - {event_name}")
    
    return all_results, all_figures

def plot_peak_aligned_to_events(peak_df, event_times, time_col='time', 
                                pre_time=4, post_time=10, bin_size=2,
                                dt=None, bin_factor=20,
                                title='Peak Analysis', ylabel='Amplitude'):
    """
    Plot peak timing and amplitude aligned to events.
    
    Parameters
    ----------
    peak_df : pd.DataFrame
        DataFrame indexed by time (numeric, in seconds), with columns:
        'amplitude', 'prominence'
    event_times : list or array
        Times of events to align to (same units as peak_df.index)
    time_col : str
        Name/label of the time axis (used only for labeling, not as a column)
    pre_time : float
        Time before event (seconds)
    post_time : float
        Time after event (seconds)
    bin_size : float
        Bin size for frequency/amplitude analysis (seconds)
    title : str
        Plot title
    ylabel : str
        Colorbar label for amplitude/metric
        
    Returns
    -------
    fig, axes : matplotlib.figure.Figure, np.ndarray of Axes
    """
    event_times = np.asarray(event_times)
    n_events = len(event_times)
    if n_events == 0:
        print("No events to align to")
        return None, None

    if dt is None:
        raise ValueError("dt (time_step)must be provided.")
    heat_bin_size = bin_factor * dt  # e.g. 20 samples per bin
       
    # Create figure with 2 rows
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1, 0.02],   # left wide, right narrow for colorbar
        height_ratios=[1, 1],
        wspace=0.0,
        hspace=0.25,   # you can tweak vertical spacing too
    )

    # 🔹 Share x-axis explicitly: ax_bins shares x with ax_heat
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bins = fig.add_subplot(gs[1, 0], sharex=ax_heat)
    cax     = fig.add_subplot(gs[0, 1])
    ax_trash = fig.add_subplot(gs[1, 1])
    ax_trash.axis('off')
    
    # Convert index to numpy array of times
    peak_times = peak_df.index.to_numpy()
    
    # Build bins based on bin_size (shared by heatmap & bottom panel)
    heat_bins = np.arange(-pre_time, post_time + heat_bin_size, heat_bin_size)
    n_bins = len(heat_bins) - 1
    
    # Heatmap matrix: rows = events, cols = time bins
    peak_matrix = np.full((n_events, n_bins), np.nan)
    
    
    # Collections for bottom binned plot
    all_peak_times = []
    all_peak_amps = []
    
    # Top subplot: Heatmap-like scatter of peak times vs events       
    for i, evt_time in enumerate(event_times):
        # Mask peaks for this event using the index
        mask = (peak_times >= evt_time - pre_time) & \
               (peak_times <= evt_time + post_time)
        
        if not np.any(mask):
            continue
        
        window_times = peak_times[mask]
        rel_times = window_times - evt_time
        amps = peak_df.loc[mask, 'prominence'].to_numpy()
        
        # Collect for bottom panel
        all_peak_times.extend(rel_times.tolist())
        all_peak_amps.extend(amps.tolist())
        
        for rel_t, amp in zip(rel_times, amps):
            bin_idx = np.digitize([rel_t], heat_bins)[0] - 1
            if 0 <= bin_idx < n_bins:
                if np.isnan(peak_matrix[i, bin_idx]) or amp > peak_matrix[i, bin_idx]:
                    peak_matrix[i, bin_idx] = amp
    
    # Plot heatmap
    if all_peak_times:  # at least one peak across all events
        im = ax_heat.imshow(
            peak_matrix,
            aspect='auto',
            cmap='viridis', #viridis, hot
            extent=[-pre_time, post_time, n_events - 0.5, -0.5],
            interpolation='nearest'
        )
        ax_heat.set_facecolor('white')
        
        # plt.colorbar(im, ax=ax_heat, label=ylabel)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(ylabel)
        
        ax_heat.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        # ax_heat.set_xlabel('Time relative to event (s)')
        ax_heat.set_ylabel('Trial #')
        ax_heat.set_title(f'{title} - Peak Heatmap',fontsize=14)
        
        # Set y-ticks to start from 1
        yticks = ax_heat.get_yticks()
        valid_yticks = yticks[(yticks >= 0) & (yticks < n_events)]
        ax_heat.set_yticks(valid_yticks)
        ax_heat.set_yticklabels([int(y + 1) for y in valid_yticks])
    else:
        ax_heat.text(
            0.5,
            0.5,
            "No peaks in any window",
            ha="center",
            va="center",
            transform=ax_heat.transAxes,
        )
        ax_heat.set_ylabel("Trial #")
        ax_heat.set_title(f"{title} - Peak Heatmap")
        cax.axis("off")
    
    # Bottom subplot: Binned frequency and amplitude
 
    # Create bins
    bins = np.arange(-pre_time, post_time + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate frequency (peaks per second)
    frequencies = []
    amplitudes = []
    all_peak_times_arr = np.array(all_peak_times)
    all_peak_amps_arr = np.array(all_peak_amps)
    
    for i in range(len(bins) - 1):
        mask = (all_peak_times_arr >= bins[i]) & \
               (all_peak_times_arr < bins[i + 1])
        
        n_peaks_in_bin = np.sum(mask)
        freq = n_peaks_in_bin / (n_events * bin_size)  # Hz
        frequencies.append(freq)
        
        if n_peaks_in_bin > 0:
            amps_in_bin = all_peak_amps_arr[mask]
            amplitudes.append(np.mean(amps_in_bin))
        else:
            amplitudes.append(0)
    
    # Plot frequency
    ax_freq = ax_bins
    ax_freq.bar(
        bin_centers,
        frequencies,
        width=bin_size * 0.8,
        alpha=0.6,
        color='blue',
        label='Peak Frequency'
    )
    ax_freq.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_freq.set_xlabel('Time relative to event (s)')
    ax_freq.set_ylabel('Peak Frequency (Hz)', color='blue')
    ax_freq.tick_params(axis='y', labelcolor='blue')
    ax_freq.grid(True, alpha=0.3)
    
    # Plot amplitude on secondary y-axis
    ax_amp = ax_freq.twinx()
    ax_amp.plot(
        bin_centers,
        amplitudes,
        'o-',
        color='orange',
        linewidth=2,
        markersize=6,
        label='Mean Amplitude'
    )
    ax_amp.set_ylabel('Mean Peak Amplitude', color='orange')
    ax_amp.tick_params(axis='y', labelcolor='orange')
    ax_freq.set_xlim(-pre_time, post_time)
    
    ax_freq.set_title(f'{title} - Binned Peak Analysis (bin={bin_size}s)', fontsize=14)
    # align left y-labels of heatmap + bottom-left axis
    fig.align_ylabels([ax_heat, ax_freq])
    
    # Combine legends
    # lines1, labels1 = ax_freq.get_legend_handles_labels()
    # lines2, labels2 = ax_amp.get_legend_handles_labels()
    # ax_freq.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    return fig, (ax_heat, ax_freq, cax)

def merge_figure_dicts(existing_figs, new_figs):
    """
    Merge new figures into existing figures dictionary.
    
    Parameters:
    -----------
    existing_figs : dict
        Existing figures dictionary (e.g., self.figs)
    new_figs : dict
        New figures to merge in
    
    Returns:
    --------
    dict : Merged figures dictionary
    """
    if existing_figs is None:
        return new_figs
    
    merged = existing_figs.copy()
    
    for signal_col, signal_figs in new_figs.items():
        if signal_col not in merged:
            merged[signal_col] = {}
        merged[signal_col].update(signal_figs)
    
    return merged

def flatten_defaultdict(d, parent_key='', sep='_'):
    """Flatten nested dict/defaultdict structure"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_defaultdict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_results(results, filepath, format='pickle'):
    """
    Save event-triggered analysis results to file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_signals_by_events
    filepath : str
        Path to save file (without extension for pickle/excel)
    format : str
        'pickle': Complete data (recommended for reloading)
        'excel': Summary tables (human-readable) using pyexcelerate
        'csv': Individual CSV files per signal/event
    """
    import pickle
    
    if format == 'pickle':
        # Save complete results as pickle (best for reloading)
        with open(f'{filepath}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved complete results to {filepath}.pkl")
    
    elif format == 'excel':
        # Save summary tables to Excel with multiple sheets using pyexcelerate     
        wb = Workbook()
        
        # Flatten the results dictionary
        flattened = flatten_defaultdict(results)
        
        # Separate parameters from data
        parameter_rows = []
        
        for key, value in flattened.items():
            # Handle DataFrames
            if isinstance(value, pd.DataFrame) and not value.empty:
                sheet_name = key[:31]  # Excel sheet name limit
                # 1. Get headers
                header = [value.index.name if value.index.name else 'index'] + value.columns.tolist()
                
                # 2. Reset index to turn the index into a column, then convert everything to a list
                # This is much faster than looping with .loc
                data_rows = value.reset_index().values.tolist()
                
                # 3. Combine header and data
                data = [header] + data_rows
                
                wb.new_sheet(sheet_name, data=data)
                print(f"  Added sheet: {sheet_name}")
            
            # Handle parameters (simple types)
            elif isinstance(value, (str, bool, int, float)):
                parameter_rows.append([key, str(value)])
            
            # Handle lists (e.g., signal_channels_available_signals)
            elif isinstance(value, list):
                parameter_rows.append([key, ', '.join(str(v) for v in value)])
        
        # Add Parameters sheet
        if parameter_rows:
            param_sheet_data = [['Parameter', 'Value']] + parameter_rows
            wb.new_sheet('Parameters', data=param_sheet_data)
            print(f"  Added sheet: Parameters ({len(parameter_rows)} parameters)")
        
        wb.save(f'{filepath}.xlsx')
        print(f"Saved summary to {filepath}.xlsx")
    
    elif format == 'csv':
        # Save each DataFrame as separate CSV
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # Flatten the results
        flattened = flatten_defaultdict(results)
        
        # Save parameters as CSV
        parameter_rows = []
        for key, value in flattened.items():
            if isinstance(value, (str, bool, int, float)):
                parameter_rows.append({'Parameter': key, 'Value': str(value)})
            elif isinstance(value, list):
                parameter_rows.append({'Parameter': key, 'Value': ', '.join(str(v) for v in value)})
        
        if parameter_rows:
            param_df = pd.DataFrame(parameter_rows)
            param_path = os.path.join(filepath, 'Parameters.csv')
            param_df.to_csv(param_path, index=False)
            print("  Saved Parameters.csv")
        
        # Save DataFrames
        for key, value in flattened.items():
            if isinstance(value, pd.DataFrame):
                filename = f"{key}.csv"
                full_path = os.path.join(filepath, filename)
                value.to_csv(full_path)
                print(f"  Saved {filename}")
        
        print(f"Saved CSV files to {filepath}/ directory")

def save_results_old(results, filepath, format='pickle'):
    """
    Save event-triggered analysis results to file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_signals_by_events
    filepath : str
        Path to save file (without extension for pickle/excel)
    format : str
        'pickle': Complete data (recommended for reloading)
        'excel': Summary tables (human-readable) using pyexcelerate
        'csv': Individual CSV files per signal/event
    """
    import pickle
    
    if format == 'pickle':
        # Save complete results as pickle (best for reloading)
        with open(f'{filepath}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved complete results to {filepath}.pkl")
    
    elif format == 'excel':
        # Save summary tables to Excel with multiple sheets using pyexcelerate

        
        wb = Workbook()
        
        for signal_col, signal_data in results.items():
            # Save z-score DataFrame
            if 'zscore' in signal_data:
                zscore_df = signal_data['zscore']
                sheet_name = f"{signal_col}_zscore"[:31]
                
                # Convert DataFrame to list format for pyexcelerate
                data = [zscore_df.index.name] + zscore_df.columns.tolist()
                data = [[data[0]] + data[1:]]  # Header row
                for idx_val in zscore_df.index:
                    row = [idx_val] + zscore_df.loc[idx_val].tolist()
                    data.append(row)
                
                wb.new_sheet(sheet_name, data=data)
            
            # Save event epochs DataFrames
            if 'events' in signal_data:
                event_data_dict = signal_data['events']
                
                for event_col, event_data in event_data_dict.items():
                    for event_type, epoch_df in event_data.items():
                        # Create sheet name (max 31 chars for Excel)
                        sheet_name = f"{signal_col}_{event_col}_{event_type}"[:31]
                        
                        # Convert DataFrame to list format
                        header = [epoch_df.index.name] + epoch_df.columns.tolist()
                        data = [header]
                        for idx_val in epoch_df.index:
                            row = [idx_val] + epoch_df.loc[idx_val].tolist()
                            data.append(row)
                        
                        wb.new_sheet(sheet_name, data=data)
        
        wb.save(f'{filepath}.xlsx')
        print(f"Saved summary to {filepath}.xlsx")
    
    elif format == 'csv':
        # Save each DataFrame as separate CSV
        import os
        os.makedirs(filepath, exist_ok=True)
        
        for signal_col, signal_data in results.items():
            # Save z-score
            if 'zscore' in signal_data:
                zscore_df = signal_data['zscore']
                zscore_path = os.path.join(filepath, f"{signal_col}_zscore.csv")
                zscore_df.to_csv(zscore_path)
            
            # Save event epochs
            if 'events' in signal_data:
                event_data_dict = signal_data['events']
                
                for event_col, event_data in event_data_dict.items():
                    for event_type, epoch_df in event_data.items():
                        filename = f"{signal_col}_{event_col}_{event_type}.csv"
                        full_path = os.path.join(filepath, filename)
                        epoch_df.to_csv(full_path)
        
        print(f"Saved CSV files to {filepath}/ directory")

def save_figures(figs, filepath='figures.pdf', format='pdf', dpi=300):
    """
    Save all figures to files.
    
    Parameters:
    -----------
    figs : dict
        Figure dictionary from analyze_signals_by_events
        Structure: {signal_col: {event_col: figure_handle}}
    filepath : str
        For PDF: path to single PDF file (default: 'figures.pdf')
        For other formats: folder to save individual figure files
    format : str
        File format: 'pdf', 'png', 'svg', 'jpg'
    dpi : int
        Resolution for raster formats (png, jpg)
    """
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    
    if format == 'pdf':
        # Save all figures to a single PDF
        with PdfPages(filepath) as pdf:
            for signal_col, signal_figs in figs.items():
                for fig_name, fig in signal_figs.items():
                    pdf.savefig(fig, bbox_inches='tight')
                    print(f"Added {signal_col}_{fig_name} to PDF")
        print(f"\nAll figures saved to {filepath}")
    else:
        # Save individual files
        folder = filepath if filepath.endswith('/') else filepath
        os.makedirs(folder, exist_ok=True)
        
        for signal_col, signal_figs in figs.items():
            for event_name, fig in signal_figs.items():
                filename = f"{signal_col}_{event_name}.{format}"
                file_path = os.path.join(folder, filename)
                fig.savefig(file_path, dpi=dpi, bbox_inches='tight', format=format)
                print(f"Saved {file_path}")
        
        print(f"\nAll figures saved to {folder}/ directory")

def load_results(filepath):
    """
    Load results from pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to pickle file (with or without .pkl extension)
    
    Returns:
    --------
    dict : Results dictionary
    """
    import pickle
    
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded results from {filepath}")
    return results

def save_figures_old(figs, folder='figures', format='pdf', dpi=300):
    """
    Save all figures to files.
    
    Parameters:
    -----------
    figs : dict
        Figure dictionary from analyze_signals_by_events
        Structure: {signal_col: {event_col: figure_handle}}
    folder : str
        Folder to save figures
    format : str
        File format: 'pdf', 'png', 'svg', 'jpg'
    dpi : int
        Resolution for raster formats (png, jpg)
    """
    import os
    os.makedirs(folder, exist_ok=True)
    
    for signal_col, signal_figs in figs.items():
        for event_name, fig in signal_figs.items():
            filename = f"{signal_col}_{event_name}.{format}"
            filepath = os.path.join(folder, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
            print(f"Saved {filepath}")
    
    print(f"\nAll figures saved to {folder}/ directory")

def fig_to_one_pdf(figs,filepath, filename):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(filepath, exist_ok=True)
    
    pdf_filename = os.path.join(filepath, f'{filename}_{formatted_time}.pdf')
    #pdf_filename = os.path.join(self.output_path, os.path.basename(self.filename).split('.')[0] + '.pdf')
    with PdfPages(pdf_filename) as pdf:
        for signal_col, signal_figs in figs.items():
            for event_name, fig in signal_figs.items():
                pdf.savefig(fig)

# --------- Main GUI ---------
class FPFEDSynchronizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FP_run_FED_synchronizor")
        self.root.geometry("1650x1220")
        self.root.minsize(1650, 1220)
        
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize data containers
        self.fp_files = []
        self.cam_files = []
        self.fed_l_files = []
        self.fed_r_files = []
        
        self.df_FP = pd.DataFrame()
        self.df_run = pd.DataFrame()
        self.df_run_L = pd.DataFrame()
        self.df_run_R = pd.DataFrame()
        self.df_fed_L = pd.DataFrame()
        self.df_fed_R = pd.DataFrame()
        self.df_wheel_L_indices = pd.DataFrame()
        self.df_wheel_R_indices = pd.DataFrame()
        self.df_fed_L_indices = pd.DataFrame()
        self.df_fed_R_indices = pd.DataFrame()
        
        self.NPM_dff = pd.DataFrame()
        self.NPM_zscore = pd.DataFrame()
        self.time_column = 'Time(s)'
        self.time_step = None
        self.signal_channels = []
        self.figs = {}
        self.results = {}
        self.signal_event_mapping = {}
        
        self.output_path = None
        self.filename = None
        self.cno_time = None
        self.mapping_list = []  # List of mapping configurations
        
        # Parameters
        self.flag_470 = 2
        self.flag_415 = 1
        self.polyfit_cofs = 2
        self.filter_window = 4
        self.peak_prominence = 0.5
        self.peak_bin_size = 1
        
        # Create main frame
        main_frame = tk.Frame(root, bg='white')
        # main_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Build UI
        self.setup_ui(main_frame)
        
    def log_message(self, title, message):
        """Add message to the log panel"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding based on message type
        tag_name = title.lower()
        if tag_name == 'success':
            prefix, color = "✓", 'green'
        elif tag_name == 'warning':
            prefix, color = "⚠", 'orange'
        elif tag_name == 'error':
            prefix, color = "✗", 'red'
        else:
            prefix, color = "ℹ", 'blue'
        
        time_prefix = f"[{timestamp}] "
        header_line = f"{time_prefix} {prefix} {title}\n"
        body = f"{message}\n"
        log_entry = header_line + body
        
        # Enable text widget, add message, disable again
        self.message_log.config(state='normal')
        
        # Insert with color tag
        start_index = self.message_log.index('end-1c')
        self.message_log.insert(tk.END, log_entry)

        
        # Apply color tag to the prefix
        header_offset_start = len(time_prefix)+1
        header_length = len(prefix) + 1 + len(title)
        tag_start = f"{start_index}+{header_offset_start}c"
        tag_end = f"{tag_start}+{header_length}c"
    
        self.message_log.tag_config(tag_name, foreground=color, font=('Courier', 8, 'bold'))
        self.message_log.tag_add(tag_name, start_index, tag_end)
        
        self.message_log.config(state='disabled')
        
        # Auto-scroll to bottom
        self.message_log.see(tk.END)
    
    def clear_message_log(self):
        """Clear the message log"""
        self.message_log.config(state='normal')
        self.message_log.delete('1.0', tk.END)
        self.message_log.config(state='disabled')
        # self.log_message("Log cleared", 'info')
        self.log_message("Welcome", "Choose\nOption 1: import pre-merged csv file\nOption 2: start from raw files\nReset: start over")
    
    def show_message(self, title, message, msg_type='info'):
        """Show message in both popup and log"""
        # Log the message
        self.log_message(title, message)
        
        # # Show popup
        # if msg_type == 'error':
        #     messagebox.showerror(title, message)
        # elif msg_type == 'warning':
        #     messagebox.showwarning(title, message)
        # else:
        #     messagebox.showinfo(title, message)
    
    def setup_ui(self, main_frame):
        # STEP 1: Load files
        step1_frame = tk.LabelFrame(main_frame, text="STEP 1: Load files", 
                                    font=('Arial', 12, 'bold'), 
                                    bg='white', fg='black',
                                    bd=3, relief='solid', highlightbackground='blue')
        step1_frame.place(x=5, y=5, width=1190, height=635)
        self.create_step1(step1_frame)
        
        # STEP 2: Channel Mapping
        mapping_frame = tk.LabelFrame(main_frame, text="STEP 2: Define Signal-Event Associations", 
                                      font=('Arial', 12, 'bold'), 
                                      bg='white', fg='black',
                                      bd=3, relief='solid', highlightbackground='blue')
        mapping_frame.place(x=5, y=650, width=1190, height=280)
        self.create_channel_mapping(mapping_frame)

        # STEP 3: Align to DIOS
        step3_frame = tk.LabelFrame(main_frame, text="STEP 3: Align to DIOS", 
                                    font=('Arial', 12, 'bold'), 
                                    bg='white', fg='black',
                                    bd=3, relief='solid', highlightbackground='blue')
        step3_frame.place(x=5, y=935, width=1190, height=110)
        self.create_step3(step3_frame)
        
        # STEP 4: Save Figs
        step4_frame = tk.LabelFrame(main_frame, text="STEP 4: Save Figs", 
                                    font=('Arial', 12, 'bold'), 
                                    bg='white', fg='black',
                                    bd=3, relief='solid', highlightbackground='blue')
        step4_frame.place(x=5, y=1050, width=1190, height=150)
        self.create_step4(step4_frame)
        
        message_frame = tk.Frame(main_frame, bg='white')
        message_frame.place(x=1200, y=5, width=440, height=1200)
        self.create_log_panel(message_frame)
        
   
    def create_log_panel(self, parent):      
        # --- Label at the top ---
        log_label = tk.Label(
            parent,
            text="Message:",
            font=('Arial', 10, 'bold'),
            bg='black',
            fg='white',
            anchor='w',
            padx=5
        )
        log_label.pack(side=tk.TOP, fill=tk.X)
    
         # --- Middle: frame holding Text + Scrollbar ---
        log_frame = tk.Frame(parent, bg='grey90')
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 1. Create and pack the text box to the LEFT
        self.message_log = tk.Text(
            log_frame,
            bg='lightyellow',
            fg='black',
            font=('Courier', 8),
            wrap=tk.WORD,
            state='normal'
        )
        
        # 2. Create Scrollbar (linked to Text)
        log_scrollbar = tk.Scrollbar(
            log_frame,
            orient=tk.VERTICAL,
            command=self.message_log.yview # Placeholder
        )
        
        # 3. Pack the scrollbar to the RIGHT first (otherwise can not see it)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 4. PACK TEXT SECOND
        self.message_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 5. Link them together
        self.message_log.config(yscrollcommand=log_scrollbar.set)
        # self.message_log.insert(tk.END, "Choose option 1 to import pre-merged csv file, or option 2 to start from raw files")
        self.show_message("Welcome", "Choose\nOption 1: import pre-merged csv file\nOption 2: start from raw files\nReset: start over")
        
        # --- Bottom: clear button ---
        clear_btn = tk.Button(
            parent,
            text="Clear Log",
            bg='orange',
            fg='white',
            font=('Arial', 10, 'bold'),
            command=self.clear_message_log
        )
        clear_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))
    
    def create_step1(self, parent):
        # Option 1: Load merged_file
        option1_label = tk.Label(parent, text="Option 1:", 
                                font=('Arial', 10, 'bold'), bg='red', fg='white')
        option1_label.place(x=15, y=5, width=100, height=25)
        
        load_merged_btn = tk.Button(parent, text="Load_merged_file", 
                                    bg='blue', fg='white', font=('Arial', 9, 'bold'),
                                    command=self.load_merged_file)
        load_merged_btn.place(x=120, y=5, width=200, height=25)
        
        # reset
        tk.Button(parent, text="Reset", bg='red', fg='white', 
                 font=('Arial', 12, 'bold'), command=self.reset).place(
                 x=905, y=5, width=160, height=30)
        
        # Option 2: Load raw files
        option2_frame = tk.LabelFrame(parent, text="Option 2:", 
                                    font=('Arial', 10, 'bold'), 
                                    bg='white')
        option2_frame.place(x=5, y=40, width=1175, height=550)
        self.create_option2(option2_frame)
        
    def create_option2(self, parent):
        option2_label = tk.Label(parent, text="Load raw files into lists", 
                                font=('Arial', 10, 'bold'), bg='white')
        option2_label.place(x=5, y=5, width=380, height=25)
        
        # List boxes with Add/Delete buttons
        x, y = 8, 40
        button_width = 135
        list_spacing = 290
        list_width = 280
        
        # FP List
        tk.Button(parent, text="Add_FP", bg='green', fg='white', font=('Arial', 9, 'bold'),
                 command=self.add_fp).place(x=x, y=y, width=button_width, height=30)
        tk.Button(parent, text="Delete_FP", bg='orange', fg='white', font=('Arial', 9, 'bold'),
                 command=self.delete_fp).place(x=x+list_width-button_width, y=y, width=button_width, height=30)
        self.fp_listbox = tk.Listbox(parent, bg='lightgray', selectmode=tk.EXTENDED)
        self.fp_listbox.place(x=x, y=y+35, width=list_width, height=160)
        
        # Cam List
        x += list_spacing
        tk.Button(parent, text="Add_Cam", bg='green', fg='white', font=('Arial', 9, 'bold'),
                 command=self.add_cam).place(x=x, y=y, width=button_width, height=30)
        tk.Button(parent, text="Delete_Cam", bg='orange', fg='white', font=('Arial', 9, 'bold'),
                 command=self.delete_cam).place(x=x+list_width-button_width, y=y, width=button_width, height=30)
        self.cam_listbox = tk.Listbox(parent, bg='lightgray', selectmode=tk.EXTENDED)
        self.cam_listbox.place(x=x, y=y+35, width=list_width, height=160)
        
        # FED_L List
        x += list_spacing
        tk.Button(parent, text="Add_FED_L", bg='green', fg='white', font=('Arial', 9, 'bold'),
                 command=self.add_fed_l).place(x=x, y=y, width=button_width, height=30)
        tk.Button(parent, text="Delete_FED_L", bg='orange', fg='white', font=('Arial', 9, 'bold'),
                 command=self.delete_fed_l).place(x=x+list_width-button_width, y=y, width=button_width, height=30)
        self.fed_l_listbox = tk.Listbox(parent, bg='lightgray', selectmode=tk.EXTENDED)
        self.fed_l_listbox.place(x=x, y=y+35, width=list_width, height=160)
        
        # FED_R List
        x += list_spacing
        tk.Button(parent, text="Add_FED_R", bg='green', fg='white', font=('Arial', 9, 'bold'),
                 command=self.add_fed_r).place(x=x, y=y, width=button_width, height=30)
        tk.Button(parent, text="Delete_FED_R", bg='orange', fg='white', font=('Arial', 9, 'bold'),
                 command=self.delete_fed_r).place(x=x+list_width-button_width, y=y, width=button_width, height=30)
        self.fed_r_listbox = tk.Listbox(parent, bg='lightgray', selectmode=tk.EXTENDED)
        self.fed_r_listbox.place(x=x, y=y+35, width=list_width, height=160)
        
        # Parameters section
        y = 255
        y_space = 50
        
        # Row 1: Column selections
        tk.Label(parent, text="wheel_L_col =", bg='white', font=('Arial', 9)).place(
                x=5, y=y, width=140, height=25)
        self.Wheel_left_col = tk.Entry(parent, bg='lightgray')
        self.Wheel_left_col.place(x=150, y=y, width=50, height=25)
        self.Wheel_left_col.insert(0, "8")
        
        tk.Label(parent, text="wheel_R_col =", bg='white', font=('Arial', 9)).place(
                x=205, y=y, width=140, height=25)
        self.Wheel_right_col = tk.Entry(parent, bg='lightgray')
        self.Wheel_right_col.place(x=350, y=y, width=50, height=25)
        self.Wheel_right_col.insert(0, "9")
        
        tk.Label(parent, text="wheel_Interval (s) =", bg='white', font=('Arial', 9)).place(
                x=405, y=y, width=160, height=25)
        self.Wheel_interval = tk.Entry(parent, bg='lightgray')
        self.Wheel_interval.place(x=570, y=y, width=50, height=25)
        self.Wheel_interval.insert(0, "5")
        
        tk.Label(parent, text="fed_col =", bg='white', font=('Arial', 9)).place(
                x=635, y=y, width=140, height=25)
        self.fed_col = tk.Entry(parent, bg='lightgray')
        self.fed_col.place(x=750, y=y, width=80, height=25)
        self.fed_col.insert(0, "3")
        
        tk.Button(parent, text="1. Load & Merge Files", bg='green', fg='white', 
                 font=('Arial', 10, 'bold'), command=self.load_and_merge).place(
                 x=905, y=y, width=250, height=30)
        
        # Row 2: FP parameters
        y += y_space
        tk.Frame(parent, bg="black").place(x=0, y=y-10, width=1175, height=2)
        
        tk.Label(parent, text="FP_470_Flag =", bg='white', font=('Arial', 9)).place(
                x=5, y=y, width=140, height=25)
        self.fp_470_entry = tk.Entry(parent, bg='lightgray')
        self.fp_470_entry.place(x=150, y=y, width=80, height=25)
        self.fp_470_entry.insert(0, "2")
        
        tk.Label(parent, text="Filtfilt Window =", bg='white', font=('Arial', 9)).place(
                x=305, y=y, width=130, height=25)
        self.filtfilt_entry = tk.Entry(parent, bg='lightgray')
        self.filtfilt_entry.place(x=445, y=y, width=80, height=25)
        self.filtfilt_entry.insert(0, "4")
        
        # Row 3
        y += y_space
        
        tk.Label(parent, text="FP_415_Flag =", bg='white', font=('Arial', 9)).place(
                x=5, y=y, width=140, height=25)
        self.fp_415_entry = tk.Entry(parent, bg='lightgray')
        self.fp_415_entry.place(x=150, y=y, width=80, height=25)
        self.fp_415_entry.insert(0, "1")
        
        tk.Label(parent, text="Reg_power =", bg='white', font=('Arial', 9)).place(
                x=305, y=y, width=130, height=25)
        self.reg_power_entry = tk.Entry(parent, bg='lightgray')
        self.reg_power_entry.place(x=445, y=y, width=80, height=25)
        self.reg_power_entry.insert(0, "2")
        
        tk.Button(parent, text="2. Clean FP to DFF", bg='green', fg='white', 
                 font=('Arial', 10, 'bold'), command=self.clean_fp_to_dff).place(
                 x=905, y=y, width=250, height=30)
        
        # Row 4
        y += y_space
        tk.Frame(parent, bg="black").place(x=0, y=y-10, width=1175, height=2)
        
        tk.Button(parent, text="3. Merge DIOs to DFF", bg='green', fg='white', 
                 font=('Arial', 10, 'bold'), command=self.merge_dios_to_dff).place(
                 x=905, y=y, width=250, height=30)
        
        # Row 5: CNO Time
        y += y_space
        tk.Frame(parent, bg="black").place(x=0, y=y-10, width=1175, height=2)
        
        tk.Label(parent, text="CNO Time (e.g. 13:00:00) =", bg='white', font=('Arial', 9)).place(
                x=5, y=y, width=220, height=25)
        self.cno_time_entry = tk.Entry(parent, bg='lightgray')
        self.cno_time_entry.place(x=300, y=y, width=255, height=25)
        
        tk.Button(parent, text="Optional: Align to CNO", bg='navy', fg='white', 
                 font=('Arial', 10, 'bold'), command=self.align_to_cno).place(
                 x=905, y=y, width=250, height=30)
        
        y += 40
        tk.Frame(parent, bg="red").place(x=0, y=y-10, width=1175, height=3)
        tk.Frame(parent, bg="red").place(x=0, y=y-5, width=1175, height=3)
                     
        tk.Button(parent, text="4. Save Merged as CSV", bg='red', fg='white', 
                 font=('Arial', 10, 'bold'), command=self.save_merged_csv).place(
                 x=905, y=y, width=250, height=30)
        
    def create_channel_mapping(self, parent):
        """Create the channel mapping interface"""
        # Left section: Define new mapping
        mapping_config_frame = tk.LabelFrame(parent, text="Define New Mapping", 
                                            font=('Arial', 10, 'bold'), bg='white')
        mapping_config_frame.place(x=5, y=5, width=625, height=240)
        
        # Recording name
        tk.Label(mapping_config_frame, text="Recording Name:", bg='white', font=('Arial', 9, 'bold')).place(
                x=10, y=10, width=150, height=25)
        self.recording_name_entry = tk.Entry(mapping_config_frame, bg='lightyellow', font=('Arial', 10))
        self.recording_name_entry.place(x=160, y=10, width=200, height=25)
        self.recording_name_entry.insert(0, 'M01_NAc')
        
        tk.Button(mapping_config_frame, text="Refresh Columns", bg='lightblue', fg='black', 
                 font=('Arial', 9, 'bold'), command=self.refresh_columns).place(
                 x=380, y=0, width=220, height=25)
        
        # Signal channel (single selection)
        tk.Label(mapping_config_frame, text="Signal Channel:", bg='white', font=('Arial', 9, 'bold')).place(
                x=10, y=45, width=150, height=25)
        tk.Label(mapping_config_frame, text="(Select ONE)", bg='white', font=('Arial', 8, 'italic')).place(
                x=10, y=70, width=150, height=15)
        
        self.signal_channel_listbox = tk.Listbox(mapping_config_frame, bg='lightgray', 
                                                 selectmode=tk.SINGLE, exportselection=False)
        self.signal_channel_listbox.place(x=160, y=45, width=200, height=120)
        
        signal_scrollbar = tk.Scrollbar(mapping_config_frame, command=self.signal_channel_listbox.yview)
        signal_scrollbar.place(x=345, y=45, width=15, height=120)
        self.signal_channel_listbox.config(yscrollcommand=signal_scrollbar.set)
        
        # Event columns (multiple selection)
        tk.Label(mapping_config_frame, text="Event Columns:", bg='white', font=('Arial', 9, 'bold')).place(
                x=380, y=30, width=150, height=25)
        tk.Label(mapping_config_frame, text="(Select MULTIPLE)", bg='white', font=('Arial', 8, 'italic')).place(
                x=380, y=55, width=150, height=15)
        
        self.event_columns_listbox = tk.Listbox(mapping_config_frame, bg='lightgray', 
                                                selectmode=tk.MULTIPLE, exportselection=False)
        self.event_columns_listbox.place(x=380, y=70, width=240, height=95)
        
        event_scrollbar = tk.Scrollbar(mapping_config_frame, command=self.event_columns_listbox.yview)
        event_scrollbar.place(x=605, y=70, width=15, height=95)
        self.event_columns_listbox.config(yscrollcommand=event_scrollbar.set)
        
        # Add mapping button
        tk.Button(mapping_config_frame, text="➜ Add to Mapping List", bg='navy', fg='white', 
                 font=('Arial', 11, 'bold'), command=self.add_mapping_to_list).place(
                 x=10, y=170, width=590, height=35)
        
        # Right section: Current mapping list
        tk.Label(parent, text="Current Mapping List:", bg='white', font=('Arial', 10, 'bold')).place(
                x=650, y=5, width=250, height=25)
        
        self.mapping_listbox = tk.Listbox(parent, bg='lightyellow', font=('Arial', 8))
        self.mapping_listbox.place(x=650, y=35, width=510, height=150)
        
        mapping_scrollbar = tk.Scrollbar(parent, command=self.mapping_listbox.yview)
        mapping_scrollbar.place(x=1160, y=35, width=15, height=150)
        self.mapping_listbox.config(yscrollcommand=mapping_scrollbar.set)
        
        # Management buttons
        tk.Button(parent, text="Delete Selected Mapping", bg='red', fg='white', 
                 font=('Arial', 9, 'bold'), command=self.delete_mapping).place(
                 x=650, y=200, width=250, height=30)
        
        tk.Button(parent, text="Clear All Mappings", bg='orange', fg='white', 
                 font=('Arial', 9, 'bold'), command=self.clear_all_mappings).place(
                 x=930, y=200, width=250, height=30)
        
        # Initialize mapping storage
        self.mapping_list = []  # List of dicts: {'recording': str, 'signal': str, 'events': [str]}
        
    def create_step3(self, parent):
        # Before event
        tk.Label(parent, text="Before event (s) =", bg='white', font=('Arial', 9)).place(
                x=5, y=10, width=145, height=25)
        self.before_event_entry = tk.Entry(parent, bg='lightgray')
        self.before_event_entry.place(x=155, y=10, width=80, height=25)
        self.before_event_entry.insert(0, "4")
        
        # After event
        tk.Label(parent, text="After event (s) =", bg='white', font=('Arial', 9)).place(
                x=20, y=45, width=130, height=25)
        self.after_event_entry = tk.Entry(parent, bg='lightgray')
        self.after_event_entry.place(x=155, y=45, width=80, height=25)
        self.after_event_entry.insert(0, "10")
        
        # Zscore_method
        tk.Label(parent, text="Event_Zscore_method", bg='white', font=('Arial', 9)).place(
                x=265, y=10, width=180, height=25)
        self.zscore_method = ttk.Combobox(parent, values=['none', 'full', 'baseline'])
        self.zscore_method.place(x=450, y=10, width=125, height=25)
        self.zscore_method.set('full')
        
        tk.Label(parent, text="Peak Prominence=", bg='white', font=('Arial', 9)).place(
                x=265, y=45, width=150, height=25)
        self.peak_prominence_entry = tk.Entry(parent, bg='lightgray')
        self.peak_prominence_entry.place(x=420, y=45, width=50, height=25)
        self.peak_prominence_entry.insert(0, self.peak_prominence)
        
        # Peak bin size
        tk.Label(parent, text="Peak Bin Size (s) =", bg='white', font=('Arial', 9)).place(
                x=500, y=45, width=155, height=25)
        self.peak_bin_size_entry = tk.Entry(parent, bg='lightgray')
        self.peak_bin_size_entry.place(x=660, y=45, width=50, height=25)
        self.peak_bin_size_entry.insert(0, self.peak_bin_size)        
        
        
        # Limit to Available checkbox
        self.limit_available_var = tk.BooleanVar()
        tk.Checkbutton(parent, text="Limit to duration", variable=self.limit_available_var,
                      bg='white', font=('Arial', 9)).place(x=750, y=10, width=160, height=25)
        
        # Limit to Available checkbox
        self.find_peak_var = tk.BooleanVar()
        tk.Checkbutton(parent, text="peak analysis", variable=self.find_peak_var,
                      bg='white', font=('Arial', 9)).place(x=750, y=45, width=160, height=25)
        
        # Align and Plot button
        tk.Button(parent, text="Align and Plot", bg='blue', fg='white', 
                 font=('Arial', 11, 'bold'), command=self.align_and_plot).place(
                 x=905, y=35, width=250, height=30)
        
    def create_step4(self, parent):
        # Path entry box
        tk.Label(parent, text="Output_Path:", bg='white', font=('Arial', 10)).place(
                x=5, y=5, width=120, height=30)
        
        self.save_path_entry = tk.Entry(parent, bg='white', relief='solid', borderwidth=2)
        self.save_path_entry.place(x=5, y=40, width=800, height=40)
        
        # Save Figures as PDF button
        tk.Button(parent, text="Save Figures as PDF", bg='blue', fg='white', 
                 font=('Arial', 11, 'bold'), command=self.save_figures_pdf).place(
                 x=905, y=15, width=250, height=30)
        
        # save results
        tk.Button(parent, text="Save Results as excel", bg='red', fg='white', 
                 font=('Arial', 11, 'bold'), command=self.save_results_file).place(
                 x=905, y=65, width=250, height=30)
                     
        # Reset button
        # tk.Button(parent, text="Reset", bg='red', fg='white', 
        #          font=('Arial', 12, 'bold'), command=self.reset).place(
        #          x=905, y=65, width=160, height=45)
    
    # ============== File Management Functions ==============
    
    def load_merged_file(self):
        """Load a pre-merged NPM CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select pre-merged NPM CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.NPM_dff = pd.read_csv(file_path)
            self.time_step = float(round(self.NPM_dff[self.time_column].diff().mean(),6))
            self.filename = os.path.basename(file_path)
            self.output_path = os.path.join(os.path.dirname(file_path), "Output")
            os.makedirs(self.output_path, exist_ok=True)
            
            # Extract channel lists
            cols = self.NPM_dff.columns
            self.signal_channels = [c for c in cols if str(c).startswith("G")]
            DIO_channels = [c for c in cols if str(c).startswith("DIO")]
            
            self.save_path_entry.delete(0, tk.END)
            self.save_path_entry.insert(0, self.output_path)
            
            # Refresh column list
            self.refresh_columns()
            
            self.show_message(
                "Success",
                ("Loaded merged FP-events CSV:\n"
                f"file_name: {self.filename}\n"
                f"Signal channels: {self.signal_channels}\n"
                f"DIO channels: {DIO_channels}\n"
                f"Time_step: {self.time_step}")
                )
            
        except Exception as e:
            self.show_message("Error", f"Failed to load merged CSV file: {e}")
    
    def add_fp(self):
        """Add FP files to the list"""
        files = filedialog.askopenfilenames(title="Select FP Files", 
                                           filetypes=[("CSV files", "*.csv")])
        for f in files:
            self.fp_files.append(Path(f))
            self.fp_listbox.insert(tk.END, Path(f).name)
    
    def delete_fp(self):
        """Delete selected FP files from list"""
        selection = self.fp_listbox.curselection()
        for idx in reversed(selection):
            del self.fp_files[idx]
            self.fp_listbox.delete(idx)
    
    def add_cam(self):
        """Add Cam files to the list"""
        files = filedialog.askopenfilenames(title="Select Cam Files",
                                           filetypes=[("CSV files", "*.csv")])
        for f in files:
            self.cam_files.append(Path(f))
            self.cam_listbox.insert(tk.END, Path(f).name)
    
    def delete_cam(self):
        """Delete selected Cam files from list"""
        selection = self.cam_listbox.curselection()
        for idx in reversed(selection):
            del self.cam_files[idx]
            self.cam_listbox.delete(idx)
    
    def add_fed_l(self):
        """Add FED_L files to the list"""
        files = filedialog.askopenfilenames(title="Select FED_L Files",
                                           filetypes=[("CSV files", "*.csv")])
        for f in files:
            self.fed_l_files.append(Path(f))
            self.fed_l_listbox.insert(tk.END, Path(f).name)
    
    def delete_fed_l(self):
        """Delete selected FED_L files from list"""
        selection = self.fed_l_listbox.curselection()
        for idx in reversed(selection):
            del self.fed_l_files[idx]
            self.fed_l_listbox.delete(idx)
    
    def add_fed_r(self):
        """Add FED_R files to the list"""
        files = filedialog.askopenfilenames(title="Select FED_R Files",
                                           filetypes=[("CSV files", "*.csv")])
        for f in files:
            self.fed_r_files.append(Path(f))
            self.fed_r_listbox.insert(tk.END, Path(f).name)
    
    def delete_fed_r(self):
        """Delete selected FED_R files from list"""
        selection = self.fed_r_listbox.curselection()
        for idx in reversed(selection):
            del self.fed_r_files[idx]
            self.fed_r_listbox.delete(idx)
    
    def load_and_merge(self):
        """Step 1: Load and merge files"""
        try:
            # Load files
            self.df_FP = load_merge_sort(self.fp_files)
            self.df_run = load_merge_sort(self.cam_files)
            self.df_fed_L = load_merge_sort(self.fed_l_files)
            self.df_fed_R = load_merge_sort(self.fed_r_files)
            
            # Get column numbers
            wheel_L_col = int(self.Wheel_left_col.get())
            wheel_R_col = int(self.Wheel_right_col.get())
            fed_col = int(self.fed_col.get())
            
            # Store or export as needed
            self.show_message("Success",
                          ("Files loaded and merged successfully!\n"
                           f"FP loaded: {self.df_FP.shape}\n"
                           f"CamFP loaded: {self.df_run.shape}\n"
                           f"FED_L loaded: {self.df_fed_L.shape}\n"
                           f"FED_R loaded: {self.df_fed_R.shape}\n")
                          )
            
            #retrend_event
            self.df_run = self.retrend_event(self.df_run,[wheel_L_col, wheel_R_col])
            self.df_fed_L = self.retrend_event(self.df_fed_L, fed_col)
            self.df_fed_R = self.retrend_event(self.df_fed_R, fed_col)
            
            # creat a Time_s col (self.time_column) if df exist
            self.df_FP[self.time_column] = self.df_FP['ComputerTimestamp']/1000
            
            for df in [self.df_run, self.df_fed_L, self.df_fed_R]:
                if not df.empty:
                    df[self.time_column] = df.iloc[:, 1] / 1000
            self.show_message("Success", "All dataframes have Time_s column at the end now")      
            
            if not self.df_run.empty:
                self.df_run_L = self.df_run.iloc[:, [-1, wheel_L_col]]
                self.df_run_R = self.df_run.iloc[:, [-1, wheel_R_col]]
            
            # get wheel on/off indices (ignore 1-turn run)
            interval = float(self.Wheel_interval.get())
            self.df_wheel_L_indices = self.get_wheel_indices(self.df_run_L, interval, 'DIO_wheel_L')
            self.df_wheel_R_indices = self.get_wheel_indices(self.df_run_R, interval, 'DIO_wheel_R')
            
            # get feed on time with cum number
            if not self.df_fed_L.empty:
                self.df_fed_L_indices = self.get_fed_indices(self.df_fed_L, fed_col, 'DIO_fed_L')
            if not self.df_fed_R.empty:              
                self.df_fed_R_indices = self.get_fed_indices(self.df_fed_R, fed_col, 'DIO_fed_R')     
                
            # Set output path
            if self.fp_files:
                self.filename = self.fp_files[0].name
                self.output_path = os.path.join(self.fp_files[0].parent, "Output")
                os.makedirs(self.output_path, exist_ok=True)
                self.save_path_entry.delete(0, tk.END)
                self.save_path_entry.insert(0, self.output_path)
            self.show_message("1. Load and Merge", "Successfully loading and merging all the files")
            
        except Exception as e:
            self.show_message("Error", f"Failed to load: {str(e)}")
                
    def retrend_event(self, df, col_list):
        """
        col_list: either a single int or a list/tuple of int column indices (0-based)..
        
        For each column index in col_list:
          - find rows where that column's value decreases (diff < 0)
          - from each such row until the next decrease (or end of df),
            add the previous row's value (re-trend after a reset).
        Modifies df in place and returns it.
        """
        
        # If df is empty, nothing to do
        # ---- 1. Handle df being None or wrong type ----
        if df is None or not isinstance(df, pd.DataFrame):
            # Just return whatever we got; caller can decide what to do
            # print('None')
            return df

        if isinstance(col_list, int):
            col_list = [col_list]
        elif not col_list:  # None or empty list/tuple
            # print('Not list')
            return df
       
        # ---- 3. If df is empty, nothing to do ----
        if df.empty:
            # print('Empty')
            return df

        df = df.dropna(subset=[df.columns[c] for c in col_list]).reset_index(drop=True)
        n_cols = df.shape[1]

        for col in col_list:
            # skip invalid column indices
            if col < 0 or col >= n_cols:
                self.show_message("Warning", f"Column index {col} out of range (0–{n_cols-1}), skipped.")
                continue
            
            s = df.iloc[:, col]
            # positions where the diff is negative
            mask = s.diff().fillna(0) < 0
            bad_pos = list(mask.to_numpy().nonzero()[0])  # integer positions

            if not bad_pos:
                self.show_message("Warning", f"Don't found not sorted {df.columns[col]}.")
                continue

            self.show_message("Unsorted", f'Found not sorted {df.columns[col]}. Fix now...')
            # add sentinel "end" position
            bad_pos.append(len(df))

            for i in range(len(bad_pos) - 1):
               start = bad_pos[i]
               end = bad_pos[i + 1]  # exclusive

               # If start is 0, there's no previous row to reference
               if start == 0:
                   continue

               # add previous row's value to the segment [start:end)
               df.iloc[start:end, col] += df.iloc[start - 1, col]

            self.show_message("Success", f'Found not sorted {df.columns[col]}. Fix now...')        
            return df
    
    def get_wheel_indices(self, df_run, interval, DIO_lable):
        print('working on this: on/off wheel PAIR based on interval, return self.df_DIO_run')
        #??? call add_Bonsai_DIO at the end
        # ---- Helper: interval filter for a single DIO column ----
        def filter_events(df_run, time_col, dio_col, interval):
            """
            Filter DIO events and encode them as:
                - first event           -> 1
                - last event            -> 0
                - middle events:
                      cond_prev only    -> 1
                      cond_next only    -> 0
                Rules:
                - if there is exactly ONE event: return (t,1) and (t+1,0)
                - when appending a new '1':
                    if the previous appended dio was also 1,
                    first insert (previous_time + 1, 0)
                - at the end, ensure even length (1–0 pairs) by dropping last if odd.
            """
        
            # 0 events → return empty df
            df = df_run.copy()
            df[dio_col] = (df[dio_col].diff().fillna(0) == 1).astype(int) #fillna for first row
            
            events = df[df[dio_col] == 1].copy()
            events.head()
            if events.empty:
                return pd.DataFrame(columns=[time_col, dio_col])
        
            # 1 event → special rule
            if len(events) == 1:
                t0 = events[time_col].iloc[0]
                return pd.DataFrame({
                    time_col: [t0, t0 + interval/2],
                    DIO_lable: [1, 0]
                })
        
            # MULTIPLE EVENTS
            times = events[time_col].values
        
            # Compute diffs
            prev_raw = times[1:] - times[:-1]
            next_raw = prev_raw.copy()
        
            prev_diff = pd.Series([float('nan')] + prev_raw.tolist())
            next_diff = pd.Series(next_raw.tolist() + [float('nan')])
        
            time_values = []
            dio_values = []
            
            # 20251203: skip all only 1 turn per ride
            # use cond_prev != cond_next
            for i in range(len(times)):
                t = times[i]
        
                if i == 0:
                    if next_diff.iloc[i] < interval:
                        # First event → DIO=1
                        time_values.append(t)
                        dio_values.append(1)
        
                elif i == len(times) - 1:
                    if prev_diff.iloc[i] < interval:                    
                        time_values.append(t)
                        dio_values.append(0)     
        
                else:
                    cond_prev = prev_diff.iloc[i] > interval
                    cond_next = next_diff.iloc[i] > interval
        
                    if cond_prev != cond_next:      
                        new_dio = 1 if cond_prev else 0  # cond_prev takes priority 

                        time_values.append(t)
                        dio_values.append(new_dio)
        
            # ensure even length (1–0 pairs): if odd, drop last
            # if len(dio_values) % 2 == 1:
            #     time_values = time_values[:-1]
            #     dio_values = dio_values[:-1]
        
            return pd.DataFrame({
                time_col: time_values,
                DIO_lable: dio_values
            })

        if df_run.empty:
            return df_run
        # ---- 3 & 4. Filter events for 2nd and 3rd columns ----
        time_col_name = self.time_column
        dio1_name = df_run.columns[1]
        
        df_run_indice = filter_events(df_run, time_col_name, dio1_name, interval) # todo: replace 1000ms with value from gui
        
        # ---- 2 & 5. Plot and stack filtered events ----
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 6))

        # Top subplot: 2nd col vs time
        ax.step(df_run[time_col_name], df_run[dio1_name], where='post', color='gray')
        ax.set_ylabel(dio1_name)

        # Overlay filtered events on top subplot
        if not df_run_indice.empty:
            for t, v in df_run_indice.itertuples(index=False, name=None):
                color = 'blue' if v == 1 else 'red'
                ax.axvline(t, color=color, linewidth=1.5)

        plt.tight_layout()
        self.figs[dio1_name] = fig
        plt.show()

        return df_run_indice
        
    def get_fed_indices(self, df_fed, event_col_num, DIO_lable):
        print('working on this: times eat pellet, return self.df_DIO_fed')
        # problem: only on, not good for add_Bonsai_DIO
        #??? Solve: merge directly with df_L
        # Get column names for readability
        time_col = self.time_column   # 2nd column
        dio_col  = df_fed.columns[event_col_num]   # 4th column
        
        # Step 1: rows where the 4th column has a rising edge (diff == 1)
        mask = df_fed[dio_col].diff().fillna(0) == 1
        
        # Step 2: build df_fed_indices
        #   - time = 2nd column - 200
        #   - value = 4th column (at those rising edges)
        df_fed_indices = pd.DataFrame({
            time_col: df_fed.loc[mask, time_col] - 200, # when record, we count on the end of 200ms pulse
            DIO_lable:  df_fed.loc[mask, dio_col]
        }).reset_index(drop=True)
        
        return df_fed_indices
    
    def clean_fp_to_dff(self):
        """Step 2: Clean FP to DFF"""
        try:
            if self.df_FP.empty:
                self.show_message("No Data", "Please load FP files first!")
                return
            
            # Get parameters
            self.flag_470 = int(self.fp_470_entry.get())
            self.flag_415 = int(self.fp_415_entry.get())
            self.filter_window = int(self.filtfilt_entry.get())
            self.polyfit_cofs = int(self.reg_power_entry.get())
            
            # Get signal columns
            signal_col_names = self.df_FP.columns[self.df_FP.columns.str.startswith("G")].tolist()
            self.signal_channels = signal_col_names
            
            # Filter data based on LedState
            df_signal = self.df_FP.loc[self.df_FP['LedState'] == self.flag_470, 
                                      [self.time_column, *signal_col_names]].reset_index(drop=True)
            
            df_refs = self.df_FP.loc[self.df_FP['LedState'] == self.flag_415, 
                                    signal_col_names].reset_index(drop=True)
            
            # Calibrate references
            df_refs_calibrated = df_refs.rolling(window=2, min_periods=1).mean()
            
            # Trim to match
            min_length = min(len(df_signal), len(df_refs_calibrated))
            df_signal = df_signal.iloc[:min_length]
            df_refs_calibrated = df_refs_calibrated.iloc[:min_length]
            
            # Filter signals
            for col in signal_col_names:
                df_signal[col] = self.filterSignal(df_signal[col], self.filter_window)
                df_refs_calibrated[col] = self.filterSignal(df_refs_calibrated[col], self.filter_window)
            
            # Calculate dF/F
            self.NPM_dff = df_signal.copy()
            self.time_step = float(round(self.NPM_dff[self.time_column].diff().mean(), 6))
            
            for col in signal_col_names:
                polyfit_cofs = np.polyfit(df_refs_calibrated[col], df_signal[col], self.polyfit_cofs)
                fit_ref = np.polyval(polyfit_cofs, df_refs_calibrated[col])
                self.NPM_dff[col] = (df_signal[col] - fit_ref) / fit_ref
            
            # Calculate z-score
            self.NPM_zscore = self.NPM_dff.copy()
            self.NPM_zscore[signal_col_names] = self.NPM_zscore[signal_col_names].apply(zscore)
            
            # Refresh column list
            self.refresh_columns()
            
            self.show_message("2. clean FP to DFF", "Successfully cleaned and converted FP to dF/F!")
            
        except Exception as e:
            self.show_message("Error", f"Failed to clean FP: {str(e)}\n{traceback.format_exc()}")
    
    def filterSignal(self, data, filter_window):
        """Apply filtfilt to signal"""
        if filter_window == 0 or filter_window == 1:
            return data
        else:
            b = np.divide(np.ones((filter_window,)), filter_window)
            a = 1
            return signal.filtfilt(b, a, data)
    
    def merge_dios_to_dff(self):
        """Step 3: Merge DIOs to DFF"""
        try:
            if self.NPM_dff.empty:
                self.show_message("Warning", "No NPM_dff found. Please clean FP to DFF first!")
                return
            
            # Merge DIO indices
            dfs = [self.df_wheel_L_indices, self.df_wheel_R_indices,
                  self.df_fed_L_indices, self.df_fed_R_indices]
            dfs = [df for df in dfs if df is not None and not df.empty]
            
            for df in dfs:
                self.add_bonsai_DIO(self.NPM_dff, df)
            
            # Refresh columns
            self.refresh_columns()
            
            self.show_message("3. Merge DIOs to DFF", "Successfully merged DIOs to DFF dataframe!")
            
        except Exception as e:
            self.show_message("Error", f"Failed to merge DIOs: {str(e)}\n{traceback.format_exc()}")
    
    def add_bonsai_DIO(self, df, df_DIO, DIOs=None):
        """Add DIO columns to dataframe"""
        time_column = self.time_column
        
        if DIOs is None or len(DIOs) == 0:
            DIOs = df_DIO.columns[1:]
        
        df[DIOs] = 0
        n = len(df_DIO)
        
        if n == 0:
            return df
        
        if n == 1:
            row = df_DIO.iloc[0]
            start_time = row[time_column]
            mask = (df[time_column] >= start_time)
            df.loc[mask, DIOs] = row[DIOs].values
            return df
        
        for i in range(n - 1):
            start_time = df_DIO.at[df_DIO.index[i], time_column]
            end_time = df_DIO.at[df_DIO.index[i + 1], time_column]
            mask = (df[time_column] >= start_time) & (df[time_column] < end_time)
            df.loc[mask, DIOs] = df_DIO.loc[df_DIO.index[i], DIOs].values
        
        return df
    
    def align_to_cno(self):
        """Optional: Align to CNO time"""
        try:
            cno_time_str = self.cno_time_entry.get().strip()
            if not cno_time_str:
                self.show_message("Warning", "Please enter CNO time and load cam files!")
                return
            
            td = pd.to_timedelta(cno_time_str)
            self.cno_time = td.total_seconds() # conver datetime into sec
                       
            # Align all dataframes
            dfs = [self.NPM_dff, self.df_run, self.df_run_L, self.df_run_R, 
                  self.df_fed_L, self.df_fed_R, self.df_wheel_L_indices,
                  self.df_wheel_R_indices, self.df_fed_L_indices, self.df_fed_R_indices]
            
            for df in dfs:
                if df is not None and not df.empty:
                    df[self.time_column] = df[self.time_column] - self.cno_time #all in sec
            
            self.show_message("Optional: Align to CNO time", f"Successfully Aligned to CNO time at {self.cno_time:.3f}s")
            
        except Exception as e:
            self.show_message("Error", f"Failed to align: {str(e)}")

    def align_to_cno_old(self):
        """Optional: Align to CNO time"""
        try:
            cno_time_str = self.cno_time_entry.get().strip()
            if not cno_time_str or self.df_run.empty:
                self.show_message("Warning", "Please enter CNO time and load cam files!")
                return
            
            cno_time = pd.to_datetime(cno_time_str).time()
            timestamp_col = pd.to_datetime(self.df_run.iloc[:, 0])
            closest_idx = (timestamp_col.dt.time.apply(
                lambda t: abs(datetime.combine(datetime.min, t) - 
                            datetime.combine(datetime.min, cno_time)))).idxmin()
            
            self.cno_time = self.df_run.iloc[closest_idx, -1]
            
            # Align all dataframes
            dfs = [self.NPM_dff, self.df_run, self.df_run_L, self.df_run_R, 
                  self.df_fed_L, self.df_fed_R, self.df_wheel_L_indices,
                  self.df_wheel_R_indices, self.df_fed_L_indices, self.df_fed_R_indices]
            
            for df in dfs:
                if df is not None and not df.empty:
                    df[self.time_column] = df[self.time_column] - self.cno_time
            
            self.show_message("Success", f"Aligned to CNO time at {self.cno_time:.3f}s")
            
        except Exception as e:
            self.show_message("Error", f"Failed to align: {str(e)}")
    
    # ============== Channel Mapping Functions ==============
    
    def refresh_columns(self):
        """Refresh the available columns list from loaded data"""
        # Clear both listboxes
        self.signal_channel_listbox.delete(0, tk.END)
        self.event_columns_listbox.delete(0, tk.END)
        
        if not self.NPM_dff.empty:
            cols = self.NPM_dff.columns.tolist()
            
            # Add signal channels (G0, G1, etc.)
            signal_cols = [c for c in cols if str(c).startswith("G")]
            for col in signal_cols:
                self.signal_channel_listbox.insert(tk.END, col)
            
            # Add DIO columns
            dio_cols = [c for c in cols if str(c).startswith("DIO")]
            for col in dio_cols:
                self.event_columns_listbox.insert(tk.END, col)
            
            self.show_message("Refreshed", 
                               f"Found {len(signal_cols)} signal channels and {len(dio_cols)} DIO channels")
    
    def add_mapping_to_list(self):
        """Add current mapping configuration to the list"""
        # Get recording name
        recording_name = self.recording_name_entry.get().strip()
        if not recording_name:
            self.show_message("Warning", "Missing Info. Please enter a recording name!")
            return
        
        # Get selected signal channel (single selection)
        signal_selection = self.signal_channel_listbox.curselection()
        if not signal_selection:
            self.show_message("Warning","Missing Info. Please select one signal channel!")
            return
        
        signal_channel = self.signal_channel_listbox.get(signal_selection[0])
        
        # Get selected event columns (multiple selection)
        event_indices = self.event_columns_listbox.curselection()
        if not event_indices:
            self.show_message("Warning", "Missing Info. Please select at least one event column!")
            return
        
        event_columns = [self.event_columns_listbox.get(idx) for idx in event_indices]
        
        # Create mapping entry
        mapping_entry = {
            'recording': recording_name,
            'signal': signal_channel,
            'events': event_columns
        }
        
        # Add to list
        self.mapping_list.append(mapping_entry)
        
        # Update display
        self.update_mapping_display()
        
        self.show_message("Success", 
                           f"Added mapping:\n{recording_name} | {signal_channel} | {', '.join(event_columns)}")
    
    def delete_mapping(self):
        """Delete selected mapping from the list"""
        selected_indices = self.mapping_listbox.curselection()
        
        if not selected_indices:
            self.show_message("Warning", "No Selection. Please select a mapping to delete!")
            return
        
        # Delete in reverse order to maintain indices
        for idx in reversed(selected_indices):
            del self.mapping_list[idx]
        
        self.update_mapping_display()
        # messagebox.showinfo("Deleted", "Selected mapping(s) deleted")
    
    def clear_all_mappings(self):
        """Clear all mappings"""
        if messagebox.askyesno("Confirm", "Clear all mappings?"):
            self.mapping_list = []
            self.update_mapping_display()
    
    def update_mapping_display(self):
        """Update the mapping listbox display"""
        self.mapping_listbox.delete(0, tk.END)
        
        for i, mapping in enumerate(self.mapping_list):
            recording = mapping['recording']
            signal = mapping['signal']
            events = ', '.join(mapping['events'])
            display_text = f"[{i+1}] {recording} | Signal: {signal} | Events: {events}"
            self.mapping_listbox.insert(tk.END, display_text)
    
    def get_signal_event_mapping(self):
        """Get the current signal-event mapping for use in analysis"""
        return self.mapping_list
    
    def align_and_plot(self):
        """Step 3: Align to DIOS and plot"""
        try:
            if self.NPM_dff.empty:
                self.show_message("Warning", "No Data. Please load and process data first!")
                return
            
            if not self.mapping_list:
                self.show_message("Warning", "No Mapping. Please add at least one channel mapping!")
                return
            
            # Get parameters
            before_event = float(self.before_event_entry.get())
            after_event = float(self.after_event_entry.get())
            zscore_method = self.zscore_method.get()
            limit_available = self.limit_available_var.get()
            
            labels = {'none': 'RAW ΔF/F',
                     'full': 'Z-score (full trace)',
                     'baseline': 'Z-score (baseline)'}
            
            # Close existing figures before creating new ones
            if self.figs:
                for signal_col, signal_figs in list(self.figs.items()): #use list to avoid crash after deletion
                    if isinstance(signal_figs, dict):
                        for fig_name, fig in signal_figs.items():
                            plt.close(fig)
                            self.show_message("Success",f"Closed nested figure: {fig_name} from {signal_col}")
                        del self.figs[signal_col]
            
            # self.figs = {}  # Reset figures
            
            # Process each mapping
            for mapping in self.mapping_list:
                recording_name = mapping['recording']
                signal_channel = mapping['signal']  # Now single string instead of list
                event_columns = mapping['events']
                
                self.show_message("Alignment",
                         (f"Processing: {recording_name}\n"
                          f"Signal: {signal_channel}\n"
                          f"Events: {event_columns}\n")
                          )
                
                # Check if signal exists
                if signal_channel not in self.NPM_dff.columns:
                    self.show_message("Warning", f"Signal {signal_channel} not found in data")
                    continue
                
                # Use custom analyze function
                results, new_figs = self.analyze_with_mapping(
                    recording_name=recording_name,
                    signal_col=signal_channel,
                    event_cols=event_columns,
                    pre_time=before_event,
                    post_time=after_event,
                    zscore_method=zscore_method,
                    limit_available=limit_available,
                    ylabel=labels[zscore_method]
                )
                
                # Merge figures
                self.figs = merge_figure_dicts(self.figs, new_figs)
                
                # Store results with recording name
                result_key = f"{recording_name}_{signal_channel}"
                if result_key not in self.results:
                    self.results[result_key] = {}
                self.results[result_key].update(results)
            
            self.show_message("Success", 
                              f"Event alignment and plotting completed!\n"
                              f"Processed {len(self.mapping_list)} mapping(s)\n"
                              f"Generated {sum(len(new_figs) for figs in self.figs.values())} figure(s)")
            
        except Exception as e:
            self.show_message("Error", f"Failed to align and plot: {str(e)}\n{traceback.format_exc()}")
    
    def analyze_with_mapping(self, recording_name, signal_col, event_cols, pre_time, post_time,
                            zscore_method, limit_available, ylabel):
        """Analyze signal with custom event mapping and recording name"""
        from scipy import stats as scipy_stats
        
        results = {}
        figures = {}
        peak_df = pd.DataFrame()
        
        ts_data = self.NPM_dff
        time_col = self.time_column
        dt = float(round(ts_data[time_col].diff().mean(), 6))
        
        # Calculate z-score
        signal_zscore = scipy_stats.zscore(ts_data[signal_col].values)
        zscore_df = pd.DataFrame({
            time_col: ts_data[time_col].values,
            'zscore': signal_zscore
        })
        zscore_df = zscore_df.set_index(time_col)
        
        results['zscore'] = zscore_df
        
        # find peak with zscore
        if self.find_peak_var.get():
            self.peak_prominence = float(self.peak_prominence_entry.get())
            self.peak_bin_size = float(self.peak_bin_size_entry.get())
            # Find peaks in z-scored data
            peak_df, figures['peak_check'] = self.find_peaks_in_signals(zscore_df, signal_col)
            results['peak'] = peak_df
            
            self.show_message("Success", f"Peak Finder: found {len(results['peak'])} peaks in {signal_col} (prominence >= {self.peak_prominence})")
        
        # Add zscore column if needed
        zscore_col_name = f'{signal_col}_zscore'
        if zscore_method == 'full':
            ts_data_with_zscore = ts_data.copy()
            ts_data_with_zscore[zscore_col_name] = signal_zscore
        else:
            ts_data_with_zscore = ts_data
        
        # Plot trace with events
        trace_fig, _ = plot_signal_with_events(
            ts_data, signal_col=signal_col, event_cols=event_cols,
            time_col=time_col
        )
        # Add recording name as suptitle
        trace_fig.suptitle(f"{recording_name} - {signal_col}", fontsize=14, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        figures['trace'] = trace_fig
        
        # Extract epochs
        epoch_results = extract_event_epochs_from_columns(
            ts_data_with_zscore, event_columns=event_cols,
            time_col=time_col, value_col=signal_col,
            pre_time=pre_time, post_time=post_time,
            limit_to_available=limit_available,
            zscore_method=zscore_method,
            zscore_col=zscore_col_name if zscore_method == 'full' else None
        )
        
        results.update(epoch_results)
        
        # Plot event-triggered averages
        for event_name, event_dict in epoch_results.items(): #event_name is event_col name
            if len(event_dict) > 0:
                fig, _ = plot_event_triggered_average(
                    event_dict, title=f'{signal_col} - {event_name}',
                    ylabel=ylabel
                )
                # Add recording name as suptitle
                fig.suptitle(f"{recording_name} - {signal_col}", fontsize=14, fontweight='bold', y=0.998)
                plt.tight_layout(rect=[0, 0, 1, 0.99])
                figures[event_name] = fig
                
                # Plot peak-aligned analysis if peaks exist for this signal                           
                if not peak_df.empty:
                    # Get event times for each event type (onset/offset)
                    for evt_type in event_dict.keys():
                        # Extract event times from the index where data exists
                        # The event happens at time=0 in the aligned data
                        # We need the actual event times from the original data
                        
                        # Get event times by finding transitions in the DIO column
                        dio_data = ts_data[event_name].values
                        time_data = ts_data[time_col].values
                        
                        if evt_type == 'onset':
                            event_indices = np.where(np.diff(dio_data) == 1)[0] + 1
                        else:  # offset
                            event_indices = np.where(np.diff(dio_data) == -1)[0] + 1
                        
                        if len(event_indices) > 0:
                            event_times = time_data[event_indices]
                        
                        peak_fig, _ = plot_peak_aligned_to_events(
                                        peak_df, event_times,
                                        time_col=time_col,
                                        pre_time=pre_time,
                                        post_time=post_time,
                                        bin_size=self.peak_bin_size,
                                        dt=dt,
                                        bin_factor = 2, # more like the width of line
                                        title=f'{signal_col} - {event_name} {evt_type}',
                                        ylabel='Peak Amplitude (z-score)'
                                    )
                    
                    if peak_fig is not None:
                        peak_fig.suptitle(f"{recording_name} - {signal_col} - Peaks", 
                                        fontsize=14, fontweight='bold', y=0.998)
                        # plt.tight_layout(rect=[0, 0, 1, 0.99])
                        figures[f'{event_name}_{evt_type}_peaks'] = peak_fig
              
        # Create unique key for this combination
        result_key = f"{recording_name}_{signal_col}"
        
        return results, {result_key: figures}
    
    def find_peaks_in_signals(self, df, signal_col):
        """Find peaks in signal channels"""

        # y-axis: z-scored signal
        signal_data = df["zscore"].to_numpy()
        # x-axis: time from index
        time_data = df.index.to_numpy()
        
        peaks, properties = find_peaks(signal_data, prominence=self.peak_prominence)
        prominences = peak_prominences(signal_data, peaks)[0]
        
        # Extract peak times
        peak_times = time_data[peaks]
        
        # Create DataFrame with peak times as index
        peak_df = pd.DataFrame(
            {
                "amplitude": signal_data[peaks],
                "prominence": prominences,
            },
            index=pd.Index(peak_times, name=df.index.name or self.time_column),
        )
                
        # Plot peaks for verification
        fig = self.plot_peaks_verification(df, peak_df, signal_col)
        
        return peak_df, fig
    
    def plot_peaks_verification(self, df, peak_df, signal_col):
        """
        Plot z-scored signal with detected peaks for verification.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Must contain columns [self.time_column, 'zscore'].
        peak_df : pandas.DataFrame
            Must contain columns [self.time_column, 'prominence'].
            Can be empty (no peaks).
        signal_col : str
            Name of the signal/column being analyzed (for the title).
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        x_signal = df.index
        x_peaks  = peak_df.index
        
        # Plot z-score signal
        ax.plot(x_signal, 
               df['zscore'], 
               'b-', alpha=0.7, label='Z-score')
        
        # Plot detected peaks
        if not peak_df.empty:
            ax.plot(x_peaks, 
                   peak_df['amplitude'], 
                   'r*', markersize=10, label='Peaks')
        
        # Use index name if available, otherwise fallback
        x_label = df.index.name if df.index.name is not None else "Time (s)"
        ax.set_xlabel(x_label)
        ax.set_ylabel('Z-score')
        if peak_df.empty:
            ax.set_title(f"{signal_col} - Detected Peaks (n=0)")
        else:
            ax.set_title(f"{signal_col} - Detected Peaks (n={len(peak_df)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_parameters_to_results(self):
        """Save all GUI parameters to self.results['Parameters']"""
        parameters = {
            'analysis_info': {
                'filename': self.filename if self.filename else 'N/A',
                'output_path': self.output_path if self.output_path else 'N/A',
                'time_column': self.time_column,
                'cno_time': self.cno_time if self.cno_time else 'N/A'
            },
            'file_info': {
                'n_fp_files': len(self.fp_files),
                'n_cam_files': len(self.cam_files),
                'n_fed_l_files': len(self.fed_l_files),
                'n_fed_r_files': len(self.fed_r_files)
            },
            'fp_processing': {
                'flag_470': self.flag_470,
                'flag_415': self.flag_415,
                'filter_window': self.filter_window,
                'polyfit_cofs': self.polyfit_cofs,
                'peak_prominence': self.peak_prominence
            },
            'column_indices': {
                'wheel_L_col': self.Wheel_left_col.get() if hasattr(self, 'Wheel_left_col') else 'N/A',
                'wheel_R_col': self.Wheel_right_col.get() if hasattr(self, 'Wheel_right_col') else 'N/A',
                'wheel_interval': self.Wheel_interval.get() if hasattr(self, 'Wheel_interval') else 'N/A',
                'fed_col': self.fed_col.get() if hasattr(self, 'fed_col') else 'N/A'
            },
            'alignment_parameters': {
                'before_event_s': self.before_event_entry.get() if hasattr(self, 'before_event_entry') else 'N/A',
                'after_event_s': self.after_event_entry.get() if hasattr(self, 'after_event_entry') else 'N/A',
                'zscore_method': self.zscore_method.get() if hasattr(self, 'zscore_method') else 'N/A',
                'limit_to_available': self.limit_available_var.get() if hasattr(self, 'limit_available_var') else False,
                'peak_bin_size_s': self.peak_bin_size_entry.get() if hasattr(self, 'peak_bin_size_entry') else 'N/A'
            },
            'signal_channels': {
                'available_signals': self.signal_channels if self.signal_channels else []
            },
            'channel_mappings': [
                {
                    'recording': m['recording'],
                    'signal': m['signal'],
                    'events': ', '.join(m['events'])
                }
                for m in self.mapping_list
            ]
        }
        
        self.results['Parameters'] = parameters
        self.show_message("Success", "Parameters saved to results.")
    
    def save_figures_pdf(self):
        """Step 4: Save figures as PDF"""
        try:
            if not self.figs:
                self.show_message("Warning","No Figures. Please align and plot first!")
                return
            
            # Get save path
            default_name = f"{self.filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            filepath = filedialog.asksaveasfilename(
                initialdir=self.output_path,
                initialfile=default_name,
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            
            if filepath:
                save_figures(self.figs, filepath=filepath, format='pdf')
                self.save_path_entry.delete(0, tk.END)
                self.save_path_entry.insert(0, os.path.dirname(filepath))
                self.show_message("Success", f"Figures saved to:\n{filepath}")
                
        except Exception as e:
            self.show_message("Error", f"Failed to save figures: {str(e)}")
    
    def save_merged_csv(self):
        self.show_message("Saving","SAVING Merged RESULTS")

        formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        save_path = os.path.join(self.output_path, os.path.basename(self.filename).split('.')[0] + f'_clean_NPM_{formatted_time}.csv')
        self.NPM_dff.to_csv(save_path, index=False)
        
        self.show_message("Success","Merged RESULTS SAVED.")
        
        # Save results
        # save_results(results_full_z, filename = 'event_analysis_results', format='pickle')
        # save_results(results, filepath = self.output_path, filename = 'event_analysis_summary', format='excel')
        
        # Save figures as PDF
        # save_figures(figs, folder=self.output_path, format='pdf', dpi=300)
    
    def save_results_file(self, format='excel'):
        """Save results to file"""
        try:
            if not self.results:
                self.show_message("Warning", "No result. Please align and plot first!")
                return
            
            # Get save path
            ext = '.xlsx' if format == 'excel' else '.pkl'
            default_name = f"{self.filename.split('.')[0] if self.filename else 'results'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            self.save_parameters_to_results()
            
            if format == 'excel':
                filepath = filedialog.asksaveasfilename(
                    initialdir=self.output_path,
                    initialfile=default_name,
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                )
            else:
                filepath = filedialog.asksaveasfilename(
                    initialdir=self.output_path,
                    initialfile=default_name,
                    defaultextension=".pkl",
                    filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
                )
            
            if filepath:
                # Remove extension as save_results adds it
                filepath_no_ext = filepath.rsplit('.', 1)[0]
                save_results(self.results, filepath_no_ext, format=format)
                self.show_message("Success", f"Results saved as {format.upper()}", 'success')
                
        except Exception as e:
            self.show_message("Error", f"Failed to save results: {str(e)}", 'error')
    
    def reset(self):
        """Reset all fields"""
        if messagebox.askyesno("Reset", "Are you sure you want to reset all fields?"):
            # Clear file lists
            self.fp_files = []
            self.cam_files = []
            self.fed_l_files = []
            self.fed_r_files = []
            
            # Clear listboxes
            self.fp_listbox.delete(0, tk.END)
            self.cam_listbox.delete(0, tk.END)
            self.fed_l_listbox.delete(0, tk.END)
            self.fed_r_listbox.delete(0, tk.END)
            
            # Clear dataframes
            self.df_FP = pd.DataFrame()
            self.df_run = pd.DataFrame()
            self.NPM_dff = pd.DataFrame()
            self.figs = {}
            self.results = {}
            
            # Clear entries
            self.cno_time_entry.delete(0, tk.END)
            
            # Close all plots
            plt.close('all')
            # self.clear_all_mappings()
            self.mapping_list = []
            self.update_mapping_display()
            
            self.show_message("RESET","Reset completed")

if __name__ == "__main__":
    root = tk.Tk()
    app = FPFEDSynchronizerGUI(root)
    root.mainloop()