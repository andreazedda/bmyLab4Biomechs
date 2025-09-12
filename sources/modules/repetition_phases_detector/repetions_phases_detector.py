"""
    This script is used to detect stable regions in a signal based on variance.
    It applies a moving average filter to the signal and then finds regions where the variance is below a certain threshold.
    The script uses the colorama library for colored debug prints.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from colorama import init, Fore, Style
import logging
import traceback
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import interact
import dash
from dash import dcc, html
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import yaml  # Add YAML library for configuration
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Removes rows with NaN or malformed data from the dataframe.
    """
    df = df.dropna()  # Remove rows with NaN values
    df = df[df.applymap(np.isreal).all(axis=1)]  # Remove rows with non-numeric values
    return df


# Load configuration from YAML file
def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
    config_path (str): Path to the YAML configuration file.

    Returns:
    dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(Fore.GREEN + f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        print(Fore.RED + "Error loading configuration file:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

# Load configuration
config_path = "/Volumes/nvme/Github/igmSquatBiomechanics/sources/processes/ascending_descending_and_stable_regions7.yaml"
config = load_config(config_path)

# Update Config class to include all paths from the YAML file
class Config:
    def __init__(self, config):
        self.window_size = config.get('window_size', 250)
        self.variance_threshold = config.get('variance_threshold', 0.01)
        self.min_length = config.get('min_length', 100)
        self.slope_threshold = config.get('slope_threshold', 0.01)
        self.n_labels = config.get('n_labels', 10)
        self.paths = config.get('paths', {})
        self.output_path = self.paths.get('output_path', "output/")
        self.input_path = self.paths.get('input_path', "input/")
        self.file_name = self.paths.get('file_name', "signal.csv")
        self.file_path = self.paths.get('file_path', os.path.join(self.input_path, self.file_name))  # Load file_path

# Initialize configuration object
config = Config(config)

# Update OUTPUT_PATH to use the configuration
OUTPUT_PATH = config.output_path

# Initialize colorama for colored debug prints
init(autoreset=True)

def moving_average(signal, window_size):
    """
    Apply a moving average filter to a signal.

    Args:
    signal (np.array or list): Input signal.
    window_size (int): Size of the moving average window.

    Returns:
    np.array: Filtered signal.
    """
    try:
        # Ensure the signal is a NumPy array
        signal = np.array(signal, dtype=float)
        print(Fore.GREEN + f"Applying moving average filter with window size: {window_size}")
        result = np.convolve(signal, np.ones(window_size) / window_size, mode='same')  # Centered moving average
        print(Fore.GREEN + "Moving average filter applied successfully.")
        return result
    except Exception as e:
        print(Fore.RED + "Error in moving_average function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def find_stable_regions(signal, window_size, threshold, min_length):
    """
    Find stable regions in a signal based on variance.

    Args:
    signal (np.array): Input signal.
    window_size (int): Size of the window for variance calculation.
    threshold (float): Variance threshold for stability.
    min_length (int): Minimum length of stable region in samples.

    Returns:
    list: List of tuples with start and end indices of stable regions.
    """
    try:
        stable_regions = []
        start = None
        print(Fore.YELLOW + "Finding stable regions...")
        for i in range(len(signal) - window_size):
            window = signal[i:i + window_size]
            window_variance = np.var(window)
            #print(Fore.BLUE + f"Window {i} to {i + window_size} variance: {window_variance}")
            if window_variance < threshold:
                if start is None:
                    start = i + window_size // 2  # Center the stable region
                    print(Fore.CYAN + f"Starting new stable region at index {start}")
            else:
                if start is not None:
                    end = i + window_size // 2  # Center the stable region
                    if end - start >= min_length:
                        stable_regions.append((start, end))
                        print(Fore.MAGENTA + f"Stable region found from {start} to {end}")
                    start = None
        if start is not None and len(signal) - start >= min_length:
            stable_regions.append((start, len(signal)))
            print(Fore.MAGENTA + f"Stable region found from {start} to {len(signal)}")
        print(Fore.YELLOW + f"Total stable regions found: {len(stable_regions)}")
        return stable_regions
    except Exception as e:
        print(Fore.RED + "Error in find_stable_regions function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def get_unstable_regions(signal, stable_regions):
    """
    Get unstable regions starting from a stable region in a signal based on variance.

    Args:
    signal (np.array): Input signal.
    stable_regions (list): List of tuples with start and end indices of stable regions.

    Returns:
    list: List of tuples with start and end indices of unstable regions.

    """
    try:
        print(Fore.YELLOW + "Getting unstable regions...")
        # Get the indices of the stable regions
        stable_indices = []
        for start, end in stable_regions:
            stable_indices.extend(list(range(start, end + 1)))

        # Get the indices of the unstable regions they are the complement of the stable regions
        unstable_indices = set(range(len(signal))) - set(stable_indices)
        unstable_indices = list(unstable_indices)
        unstable_indices.sort()
        unstable_regions = []
        start = None
        for i in range(len(unstable_indices)):
            if start is None:
                start = unstable_indices[i]
            if i == len(unstable_indices) - 1 or unstable_indices[i] + 1 != unstable_indices[i + 1]:
                end = unstable_indices[i]
                unstable_regions.append((start, end))
                print(Fore.MAGENTA + f"Unstable region found from {start} to {end}")
                start = None
        print(Fore.YELLOW + f"Total unstable regions found: {len(unstable_regions)}")
        return unstable_regions
    except Exception as e:
        print(Fore.RED + "Error in get_unstable_regions function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def determine_if_unstable_regions_are_ascending_or_descending(signal, unstable_regions):
    """
    Determine if the unstable regions are ascending or descending based on the average slope in the interval.

    Args:
    signal (np.array): Input signal.
    unstable_regions (list): List of tuples with start and end indices of unstable regions.

    Returns:
    list: List of tuples with start and end indices of ascending regions.
    list: List of tuples with start and end indices of descending regions.

    """
    try:
        ascending_regions = []
        descending_regions = []
        print(Fore.YELLOW + "Determining if unstable regions are ascending or descending...")
        for start, end in unstable_regions:
            # Calculate the average slope in the interval
            slope = (signal[end] - signal[start]) / (end - start) if end != start else 0
            print(Fore.BLUE + f"Average slope in region from {start} to {end}: {slope}")
            # Determine if the region is ascending or descending
            if slope > 0:
                ascending_regions.append((start, end))
                print(Fore.GREEN + f"Ascending region found from {start} to {end}")
            elif slope < 0:
                descending_regions.append((start, end))
                print(Fore.GREEN + f"Descending region found from {start} to {end}")
            else:
                print(Fore.YELLOW + f"Flat region found from {start} to {end}, ignored.")
        print(Fore.YELLOW + f"Total ascending regions found: {len(ascending_regions)}")
        print(Fore.YELLOW + f"Total descending regions found: {len(descending_regions)}")
        return ascending_regions, descending_regions
    except Exception as e:
        print(Fore.RED + "Error in determine_if_unstable_regions_are_ascending_or_descending function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def find_ascent_regions(signal, window_size, threshold, min_length, stable_regions, slope_threshold):
    """Find ascending regions in the signal, excluding the stable regions.

    Args:
    signal (np.array): Input signal.
    window_size (int): Size of the window for variance calculation.
    threshold (float): Variance threshold for stability.
    min_length (int): Minimum length of stable region in samples.
    stable_regions (list): List of tuples with start and end indices of stable regions.
    slope_threshold (float): Minimum slope value to classify as ascending or descending.

    Returns:
    list: List of tuples with start and end indices of ascending regions.
    """
    try:
        ascending_regions = []
        start = None
        slopes = np.diff(signal)  # Calculate the slope between consecutive points

        print(Fore.YELLOW + "Finding ascending regions...")

        # Function to check if the index is within any stable region
        def is_in_stable_region(idx, stable_regions):
            for start_region, end_region in stable_regions:
                if start_region <= idx <= end_region:
                    return True
            return False

        for i in range(len(slopes)):
            if is_in_stable_region(i, stable_regions):
                # Reset the start if entering a stable region and there was an ascending region ongoing
                if start is not None and i - start >= min_length:
                    ascending_regions.append((start, i - 1))  # Record the ascending region
                    print(Fore.MAGENTA + f"Ascending region found from {start} to {i - 1}")
                start = None  # Reset for new ascending region search
                continue  # Skip over stable regions

            slope = slopes[i]  # Get the slope for the current point

            # Check if region is ascending (positive slope above threshold)
            if slope > slope_threshold:
                if start is None:
                    start = i  # Start of a new ascending region
                    print(Fore.CYAN + f"Starting ascending region at index {i}")
            else:
                # If slope falls below threshold and there is an ongoing ascending region
                if start is not None and i - start >= min_length:
                    ascending_regions.append((start, i - 1))  # Record the ascending region
                    print(Fore.MAGENTA + f"Ascending region found from {start} to {i - 1}")
                    start = None  # Reset to look for the next ascending region

        # If there's an ongoing ascending region at the end of the signal
        if start is not None and len(signal) - start >= min_length:
            ascending_regions.append((start, len(signal) - 1))
            print(Fore.MAGENTA + f"Ascending region found from {start} to {len(signal) - 1}")

        print(Fore.YELLOW + f"Total ascending regions found: {len(ascending_regions)}")
        return ascending_regions
    except Exception as e:
        print(Fore.RED + "Error in find_ascent_regions function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def find_descending_regions(signal, window_size, threshold, min_length, stable_regions, slope_threshold):
    """
    Find descending regions in the signal, excluding the stable regions.

    Args:
    signal (np.array): Input signal.
    window_size (int): Size of the window for variance calculation.
    threshold (float): Variance threshold for stability.
    min_length (int): Minimum length of stable region in samples.
    stable_regions (list): List of tuples with start and end indices of stable regions.
    slope_threshold (float): Minimum slope value to classify as ascending or descending.

    Returns:
    list: List of tuples with start and end indices of descending regions.
    """
    try:
        descending_regions = []
        start = None
        slopes = np.diff(signal)  # Calculate the slope between consecutive points

        print(Fore.YELLOW + "Finding descending regions...")

        # Function to check if the index is within any stable region
        def is_in_stable_region(idx, stable_regions):
            for start_region, end_region in stable_regions:
                if start_region <= idx <= end_region:
                    return True
            return False

        for i in range(len(slopes)):
            if is_in_stable_region(i, stable_regions):
                # Reset the start if entering a stable region and there was a descending region ongoing
                if start is not None and i - start >= min_length:
                    descending_regions.append((start, i - 1))  # Record the descending region
                    print(Fore.MAGENTA + f"Descending region found from {start} to {i - 1}")
                start = None  # Reset for new descending region search
                continue  # Skip over stable regions

            slope = slopes[i]  # Get the slope for the current point

            # Check if region is descending (negative slope below threshold)
            if slope < -slope_threshold:
                if start is None:
                    start = i  # Start of a new descending region
                    print(Fore.CYAN + f"Starting descending region at index {i}")
            else:
                # If slope rises above threshold and there is an ongoing descending region
                if start is not None and i - start >= min_length:
                    descending_regions.append((start, i - 1))  # Record the descending region
                    print(Fore.MAGENTA + f"Descending region found from {start} to {i - 1}")
                    start = None  # Reset to look for the next descending region

        # If there's an ongoing descending region at the end of the signal
        if start is not None and len(signal) - start >= min_length:
            descending_regions.append((start, len(signal) - 1))
            print(Fore.MAGENTA + f"Descending region found from {start} to {len(signal) - 1}")

        print(Fore.YELLOW + f"Total descending regions found: {len(descending_regions)}")
        return descending_regions
    except Exception as e:
        print(Fore.RED + "Error in find_descending_regions function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def remove_frequencies_above_threshold(signal, threshold):
    """
    Remove frequencies above a certain threshold using the Hilbert transform.

    Args:
    signal (np.array): Input signal.
    threshold (float): Threshold for removing high frequencies.

    Returns:
    np.array: Signal with high frequencies removed.
    """
    try:
        print(Fore.GREEN + f"Removing frequencies above threshold: {threshold}")
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        phase = np.angle(analytic_signal)
        filtered_signal = amplitude_envelope * (amplitude_envelope < threshold)
        print(Fore.GREEN + "Frequencies removed successfully.")
        return filtered_signal
    except Exception as e:
        print(Fore.RED + "Error in remove_frequencies_above_threshold function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def plot_signal_and_regions(original_signal, filtered_signal, timestamps, stable_regions):
    """
    Plot the original signal, filtered signal, and highlight stable regions.

    Args:
    original_signal (pd.Series): Original signal.
    filtered_signal (np.array): Filtered signal.
    timestamps (pd.DatetimeIndex): Timestamps of the signal.
    stable_regions (list): List of tuples with start and end indices of stable regions.
    """
    try:
        print(Fore.YELLOW + "Plotting signal and stable regions...")
        plt.figure(figsize=(14, 7))

        plt.plot(timestamps, original_signal, label='Original Data', alpha=0.7)
        plt.plot(timestamps, filtered_signal, label='Filtered Signal (Moving Average, 250 samples)', alpha=0.7)

        for idx, (start, end) in enumerate(stable_regions):
            plt.axvspan(timestamps[start], timestamps[end], color='yellow', alpha=0.3, label='Stable Region' if idx == 0 else "")

        plt.title('Original Signal, Filtered Signal, and All Stable Regions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        # show plot for three seconds
        
        print(Fore.GREEN + "Plotting completed successfully.")
    except Exception as e:
        print(Fore.RED + "Error in plot_signal_and_regions function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def plot_signal_and_regions_stable_and_ascending(original_signal, filtered_signal, timestamps, stable_regions, ascending_regions):
    """
    Plot the original signal, filtered signal, and highlight stable and ascending regions.

    Args:
    original_signal (pd.Series): Original signal.
    filtered_signal (np.array): Filtered signal.
    timestamps (pd.DatetimeIndex): Timestamps of the signal.
    stable_regions (list): List of tuples with start and end indices of stable regions.
    ascending_regions (list): List of tuples with start and end indices of ascending regions.
    """
    try:
        print(Fore.YELLOW + "Plotting signal with stable and ascending regions...")
        plt.figure(figsize=(14, 7))

        plt.plot(timestamps, original_signal, label='Original Data', alpha=0.7)
        plt.plot(timestamps, filtered_signal, label='Filtered Signal (Moving Average, 250 samples)', alpha=0.7)

        for idx, (start, end) in enumerate(stable_regions):
            plt.axvspan(timestamps[start], timestamps[end], color='yellow', alpha=0.3, label='Stable Region' if idx == 0 else "")

        for idx, (start, end) in enumerate(ascending_regions):
            plt.axvspan(timestamps[start], timestamps[end], color='red', alpha=0.3, label='Ascending Region' if idx == 0 else "")

        plt.title('Original Signal, Filtered Signal, Stable (Yellow), and Ascending (Red) Regions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        # plt.show()
        print(Fore.GREEN + "Plotting completed successfully.")
    except Exception as e:
        print(Fore.RED + "Error in plot_signal_and_regions_stable_and_ascending function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def plot_signal_and_regions_stable_ascending_descending(original_signal, filtered_signal, timestamps, stable_regions, ascending_regions, descending_regions):
    """
    Plot the original signal, filtered signal, and highlight stable, ascending, and descending regions.

    Args:
    original_signal (pd.Series): Original signal.
    filtered_signal (np.array): Filtered signal.
    timestamps (pd.DatetimeIndex): Timestamps of the signal.
    stable_regions (list): List of tuples with start and end indices of stable regions.
    ascending_regions (list): List of tuples with start and end indices of ascending regions.
    descending_regions (list): List of tuples with start and end indices of descending regions.
    """
    try:
        print(Fore.YELLOW + "Plotting signal with stable, ascending, and descending regions...")
        plt.figure(figsize=(14, 7))

        plt.plot(timestamps, original_signal, label='Original Data', alpha=0.7)
        plt.plot(timestamps, filtered_signal, label='Filtered Signal (Moving Average, 250 samples)', alpha=0.7)

        for idx, (start, end) in enumerate(stable_regions):
            plt.axvspan(timestamps[start], timestamps[end], color='yellow', alpha=0.3, label='Stable Region' if idx == 0 else "")

        for idx, (start, end) in enumerate(ascending_regions):
            plt.axvspan(timestamps[start], timestamps[end], color='red', alpha=0.3, label='Ascending Region' if idx == 0 else "")

        for idx, (start, end) in enumerate(descending_regions):
            plt.axvspan(timestamps[start], timestamps[end], color='blue', alpha=0.3, label='Descending Region' if idx == 0 else "")

        plt.title('Original Signal, Filtered Signal, Stable (Yellow), Ascending (Red), and Descending (Blue) Regions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_PATH + 'signal_intervals.png')
        
        print(Fore.GREEN + "Plotting completed successfully.")
    except Exception as e:
        print(Fore.RED + "Error in plot_signal_and_regions_stable_ascending_descending function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise
    
def plot_signal_and_regions_stable_ascending_descending_with_n_labels_on_x(original_signal, filtered_signal, timestamps, stable_regions, ascending_regions, descending_regions, n_labels, output_path):
    """
    Plot the original signal, filtered signal, and highlight stable, ascending, and descending regions.
    
    Args:
        original_signal (pd.Series): Original signal.
        filtered_signal (np.array): Filtered signal.
        timestamps (pd.DatetimeIndex): Timestamps of the signal.
        stable_regions (list): List of tuples with start and end indices of stable regions.
        ascending_regions (list): List of tuples with start and end indices of ascending regions.
        descending_regions (list): List of tuples with start and end indices of descending regions.
        n_labels (int): Number of vertical grid lines to plot on the x-axis.
    """
    try:
        # Generate tick positions and labels for the x-axis
        x_ticks = np.linspace(0, len(timestamps) - 1, n_labels, dtype=int)
        x_labels_seconds = [(timestamps[i] - timestamps[0]).total_seconds() for i in x_ticks]
        x_seconds_full = [(t - timestamps[0]).total_seconds() for t in timestamps]  # Full list for plotting
        
        print("Plotting signal with stable, ascending, and descending regions and n vertical grid lines on x-axis...")
        plt.figure(figsize=(14, 7))

        # Plot the original signal and filtered signal using full x_seconds
        plt.plot(x_seconds_full, original_signal, label='Original Data', alpha=0.7)
        plt.plot(x_seconds_full, filtered_signal, label='Filtered Signal (Moving Average, 250 samples)', alpha=0.7)

        # Highlight stable regions
        for idx, (start, end) in enumerate(stable_regions):
            if end >= len(timestamps):  # Check to prevent out-of-bounds error
                end = len(timestamps) - 1
            plt.axvspan(timestamps[start], timestamps[end], color='yellow', alpha=0.3)

        # Highlight ascending regions
        for idx, (start, end) in enumerate(ascending_regions):
            plt.axvspan(x_seconds_full[start], x_seconds_full[end], color='red', alpha=0.3, label='Ascending Region' if idx == 0 else "")

        # Highlight descending regions
        for idx, (start, end) in enumerate(descending_regions):
            plt.axvspan(x_seconds_full[start], x_seconds_full[end], color='blue', alpha=0.3, label='Descending Region' if idx == 0 else "")

        # Set the x-ticks at regular intervals with labels showing time in seconds (as float)
        plt.xticks(x_labels_seconds, [f'{sec:.2f}s' for sec in x_labels_seconds], rotation=45)

        plt.title('Original Signal, Filtered Signal, Stable (Yellow), Ascending (Red), and Descending (Blue) Regions')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent clipping of tick labels
        plt.savefig(output_path + 'signal_intervals.png')
        # plt.show()
        print("Plotting completed successfully.")

    except Exception as e:
        print(f"Error in plot_signal_and_regions_stable_ascending_descending_with_n_labels_on_x function: {e}")

def plot_signal_and_regions_stable_ascending_descending_with_n_labels_on_x_plotly(original_signal, filtered_signal, timestamps, stable_regions, ascending_regions, descending_regions, n_labels, output_path):
    """
    Plot the original signal, filtered signal, and highlight stable, ascending, and descending regions using Plotly.
    
    Args:
        original_signal (pd.Series): Original signal.
        filtered_signal (np.array): Filtered signal.
        timestamps (pd.DatetimeIndex): Timestamps of the signal.
        stable_regions (list): List of tuples with start and end indices of stable regions.
        ascending_regions (list): List of tuples with start and end indices of ascending regions.
        descending_regions (list): List of tuples with start and end indices of descending regions.
        n_labels (int): Number of vertical grid lines to plot on the x-axis.
    """
    try:
        # Convert timestamps to seconds relative to the first timestamp
        x_seconds_full = [(t - timestamps[0]).total_seconds() for t in timestamps]
        
        # Create x-ticks for labeling at regular intervals
        x_ticks = np.linspace(0, len(timestamps) - 1, n_labels, dtype=int)
        x_labels_seconds = [(timestamps[i] - timestamps[0]).total_seconds() for i in x_ticks]

        # Create the plotly figure
        fig = go.Figure()

        # Add the original signal
        fig.add_trace(go.Scatter(x=x_seconds_full, y=original_signal, mode='lines', name='Original Data', line=dict(width=2, color='blue')))

        # Add the filtered signal
        fig.add_trace(go.Scatter(x=x_seconds_full, y=filtered_signal, mode='lines', name='Filtered Signal', line=dict(width=2, color='green', dash='dash')))

        # Highlight stable regions
        for start, end in stable_regions:
            if end >= len(timestamps):  # Check to prevent out-of-bounds error
                end = len(timestamps) - 1
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="yellow", opacity=0.3)

        # Highlight ascending regions
        for start, end in ascending_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="red", opacity=0.3, layer="below", line_width=0, annotation_text="A", annotation_position="top left")

        # Highlight descending regions
        for start, end in descending_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="blue", opacity=0.3, layer="below", line_width=0, annotation_text="D", annotation_position="top left")

        # Set x-axis labels to show time in seconds with n_labels evenly distributed
        fig.update_layout(
            title='Original Signal, Filtered Signal, Stable, Ascending, and Descending Regions',
            xaxis_title='Time (seconds)',
            yaxis_title='Value',
            xaxis=dict(
                tickmode='array',
                tickvals=x_labels_seconds,
                ticktext=[f'{sec:.2f}s' for sec in x_labels_seconds]
            ),
            legend=dict(x=0.01, y=0.99),
            showlegend=True
        )

        fig.show()
        # save the plot
        fig.write_html(output_path + 'signal_intervals.html')

    except Exception as e:
        print(f"Error in plot_signal_and_regions_stable_ascending_descending_with_n_labels_on_x_plotly function: {e}")

def plot_signal_and_regions_stable_ascending_descending_with_n_labels_on_x_plotly_interactive(original_signal, filtered_signal, timestamps, stable_regions, ascending_regions, descending_regions, n_labels):
    """
    Plot the original signal, filtered signal, and highlight stable, ascending, and descending regions using Plotly.
    Also allows the user to interactively edit two vertical lines that define an interval.
    
    Args:
        original_signal (pd.Series): Original signal.
        filtered_signal (np.array): Filtered signal.
        timestamps (pd.DatetimeIndex): Timestamps of the signal.
        stable_regions (list): List of tuples with start and end indices of stable regions.
        ascending_regions (list): List of tuples with start and end indices of ascending regions.
        descending_regions (list): List of tuples with start and end indices of descending regions.
        n_labels (int): Number of vertical grid lines to plot on the x-axis.
    """
    try:
        # Convert timestamps to seconds relative to the first timestamp
        x_seconds_full = [(t - timestamps[0]).total_seconds() for t in timestamps]
        
        # Create x-ticks for labeling at regular intervals
        x_ticks = np.linspace(0, len(timestamps) - 1, n_labels, dtype=int)
        x_labels_seconds = [(timestamps[i] - timestamps[0]).total_seconds() for i in x_ticks]

        # Create the plotly figure
        fig = go.Figure()

        # Add the original signal
        fig.add_trace(go.Scatter(x=x_seconds_full, y=original_signal, mode='lines', name='Original Data', line=dict(width=2, color='blue')))

        # Add the filtered signal
        fig.add_trace(go.Scatter(x=x_seconds_full, y=filtered_signal, mode='lines', name='Filtered Signal', line=dict(width=2, color='green', dash='dash')))

        # Highlight stable regions
        for start, end in stable_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="yellow", opacity=0.3, layer="below", line_width=0, annotation_text="S", annotation_position="top left")

        # Highlight ascending regions
        for start, end in ascending_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="red", opacity=0.3, layer="below", line_width=0, annotation_text="A", annotation_position="top left")

        # Highlight descending regions
        for start, end in descending_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="blue", opacity=0.3, layer="below", line_width=0, annotation_text="D", annotation_position="top left")

        # Set x-axis labels to show time in seconds with n_labels evenly distributed
        fig.update_layout(
            title='Original Signal, Filtered Signal, Stable, Ascending, and Descending Regions',
            xaxis_title='Time (seconds)',
            yaxis_title='Value',
            xaxis=dict(
                tickmode='array',
                tickvals=x_labels_seconds,
                ticktext=[f'{sec:.2f}s' for sec in x_labels_seconds]
            ),
            legend=dict(x=0.01, y=0.99),
            showlegend=True
        )

        # Function to update the figure with interactively adjustable vertical lines
        def update_vlines(x0=0, x1=len(x_seconds_full) - 1):
            fig.update_layout(shapes=[
                dict(
                    type="line",
                    x0=x0, y0=min(original_signal), x1=x0, y1=max(original_signal),
                    line=dict(color="Magenta", width=3)
                ),
                dict(
                    type="line",
                    x0=x1, y0=min(original_signal), x1=x1, y1=max(original_signal),
                    line=dict(color="Magenta", width=3)
                )
            ])
            fig.show()

        # Add interactive sliders to adjust the vertical lines
        interact(update_vlines, x0=(0, len(x_seconds_full) - 1, 1), x1=(0, len(x_seconds_full) - 1, 1))

    except Exception as e:
        print(f"Error in plot_signal_and_regions_stable_ascending_descending_with_n_labels_on_x_plotly_interactive function: {e}")

def plot_signal_and_regions_with_interactive_vlines(original_signal, filtered_signal, timestamps, stable_regions, ascending_regions, descending_regions, n_labels):
    """
    Plot the original signal, filtered signal, and highlight stable, ascending, and descending regions using Plotly.
    Also allows the user to interactively edit two vertical lines that define an interval.
    
    Args:
        original_signal (pd.Series): Original signal.
        filtered_signal (np.array): Filtered signal.
        timestamps (pd.DatetimeIndex): Timestamps of the signal.
        stable_regions (list): List of tuples with start and end indices of stable regions.
        ascending_regions (list): List of tuples with start and end indices of ascending regions.
        descending_regions (list): List of tuples with start and end indices of descending regions.
        n_labels (int): Number of vertical grid lines to plot on the x-axis.
    """
    try:
        # Convert timestamps to seconds relative to the first timestamp
        x_seconds_full = [(t - timestamps[0]).total_seconds() for t in timestamps]
        
        # Create x-ticks for labeling at regular intervals
        x_ticks = np.linspace(0, len(timestamps) - 1, n_labels, dtype=int)
        x_labels_seconds = [(timestamps[i] - timestamps[0]).total_seconds() for i in x_ticks]

        # Create the plotly figure
        fig = go.Figure()

        # Add the original signal
        fig.add_trace(go.Scatter(x=x_seconds_full, y=original_signal, mode='lines', name='Original Data', line=dict(width=2, color='blue')))

        # Add the filtered signal
        fig.add_trace(go.Scatter(x=x_seconds_full, y=filtered_signal, mode='lines', name='Filtered Signal', line=dict(width=2, color='green', dash='dash')))

        # Highlight stable regions
        for start, end in stable_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="yellow", opacity=0.3, layer="below", line_width=0, annotation_text="S", annotation_position="top left")

        # Highlight ascending regions
        for start, end in ascending_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="red", opacity=0.3, layer="below", line_width=0, annotation_text="A", annotation_position="top left")

        # Highlight descending regions
        for start, end in descending_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="blue", opacity=0.3, layer="below", line_width=0, annotation_text="D", annotation_position="top left")

        # Create interactive vertical lines with sliders
        vertical_lines = [
            dict(
                type="line",
                x0=0, y0=min(original_signal), x1=0, y1=max(original_signal),
                line=dict(color="Magenta", width=3),
                name="Interactive Line 1"
            ),
            dict(
                type="line",
                x0=len(x_seconds_full)-1, y0=min(original_signal), x1=len(x_seconds_full)-1, y1=max(original_signal),
                line=dict(color="Magenta", width=3),
                name="Interactive Line 2"
            )
        ]

        fig.update_layout(
            shapes=vertical_lines,
            sliders=[
                {
                    "steps": [
                        {"method": "relayout", "label": str(i), "args": [{"shapes[0].x0": x_seconds_full[i], "shapes[0].x1": x_seconds_full[i]}]}
                        for i in range(len(x_seconds_full))
                    ],
                    "currentvalue": {"prefix": "Line 1: "}
                },
                {
                    "steps": [
                        {"method": "relayout", "label": str(i), "args": [{"shapes[1].x0": x_seconds_full[i], "shapes[1].x1": x_seconds_full[i]}]}
                        for i in range(len(x_seconds_full))
                    ],
                    "currentvalue": {"prefix": "Line 2: "}
                }
            ]
        )

        # Set x-axis labels to show time in seconds with n_labels evenly distributed
        fig.update_layout(
            title='Original Signal, Filtered Signal, Stable, Ascending, and Descending Regions',
            xaxis_title='Time (seconds)',
            yaxis_title='Value',
            xaxis=dict(
                tickmode='array',
                tickvals=x_labels_seconds,
                ticktext=[f'{sec:.2f}s' for sec in x_labels_seconds]
            ),
            legend=dict(x=0.01, y=0.99),
            showlegend=True
        )
        # save the plot
        fig.write_html(OUTPUT_PATH + 'signal_intervals.html')
        # Show the interactive plot
        fig.show()


    except Exception as e:
        print(f"Error in plot_signal_and_regions_with_interactive_vlines: {e}")

# Sample function to create the Dash app
def create_dash_app(original_signal, filtered_signal, timestamps, stable_regions, ascending_regions, descending_regions, n_labels):
    """
    Create a Dash app with an interactive plot for the signal data.
    
    Args:
        original_signal (pd.Series): Original signal.
        filtered_signal (np.array): Filtered signal.
        timestamps (pd.DatetimeIndex): Timestamps of the signal.
        stable_regions (list): List of tuples with start and end indices of stable regions.
        ascending_regions (list): List of tuples with start and end indices of ascending regions.
        descending_regions (list): List of tuples with start and end indices of descending regions.
        n_labels (int): Number of vertical grid lines to plot on the x-axis.
    """

    app = dash.Dash(__name__)

    # Convert timestamps to seconds relative to the first timestamp
    x_seconds_full = [(t - timestamps[0]).total_seconds() for t in timestamps]

    # Create x-ticks for labeling at regular intervals
    x_ticks = np.linspace(0, len(timestamps) - 1, n_labels, dtype=int)
    x_labels_seconds = [(timestamps[i] - timestamps[0]).total_seconds() for i in x_ticks]

    # Create the initial figure
    def create_figure(vline_positions):
        fig = go.Figure()

        # Add the original signal
        fig.add_trace(go.Scatter(x=x_seconds_full, y=original_signal, mode='lines', name='Original Data', line=dict(width=2, color='blue')))

        # Add the filtered signal
        fig.add_trace(go.Scatter(x=x_seconds_full, y=filtered_signal, mode='lines', name='Filtered Signal', line=dict(width=2, color='green', dash='dash')))

        # Highlight stable regions
        for start, end in stable_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="yellow", opacity=0.3, layer="below", line_width=0, annotation_text="S", annotation_position="top left")

        # Highlight ascending regions
        for start, end in ascending_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="red", opacity=0.3, layer="below", line_width=0, annotation_text="A", annotation_position="top left")

        # Highlight descending regions
        for start, end in descending_regions:
            fig.add_vrect(x0=x_seconds_full[start], x1=x_seconds_full[end], fillcolor="blue", opacity=0.3, layer="below", line_width=0, annotation_text="D", annotation_position="top left")

        # Add vertical lines based on user interaction
        for pos in vline_positions:
            fig.add_vline(x=pos, line=dict(color='black', width=2, dash='dash'))

        # Set x-axis labels to show time in seconds with n_labels evenly distributed
        fig.update_layout(
            title='Original Signal, Filtered Signal, Stable, Ascending, and Descending Regions',
            xaxis_title='Time (seconds)',
            yaxis_title='Value',
            xaxis=dict(
                tickmode='array',
                tickvals=x_labels_seconds,
                ticktext=[f'{sec:.2f}s' for sec in x_labels_seconds]
            ),
            legend=dict(x=0.01, y=0.99),
            showlegend=True
        )
        return fig

    # Layout of the Dash app
    app.layout = html.Div([
        dcc.Graph(id='signal-graph', figure=create_figure([10, 20])),  # Initial positions of vertical lines
        daq.Slider(id='vline-slider-1', min=0, max=x_seconds_full[-1], value=10, step=1, marks={i: f'{i}s' for i in range(0, int(x_seconds_full[-1]), 5)}),
        daq.Slider(id='vline-slider-2', min=0, max=x_seconds_full[-1], value=20, step=1, marks={i: f'{i}s' for i in range(0, int(x_seconds_full[-1]), 5)})
    ])

    # Callback to update the plot based on the slider values
    @app.callback(
        Output('signal-graph', 'figure'),
        [Input('vline-slider-1', 'value'),
         Input('vline-slider-2', 'value')]
    )
    def update_graph(vline_pos1, vline_pos2):
        # Update the figure with the new positions of vertical lines
        return create_figure([vline_pos1, vline_pos2])

    # Run the app
    app.run_server(debug=True)

def save_intervals_to_excel_chronologically(timestamps, stable_regions, ascending_regions, descending_regions, file_name_float, file_name_datetime):
    """
    Save the list of time differences in each interval (ascending, descending, stable) to two Excel files: one with timestamps in float (seconds) 
    and another with timestamps in real datetime format, both in chronological order.

    Args:
    timestamps (pd.DatetimeIndex): Timestamps of the signal.
    stable_regions (list): List of tuples with start and end indices of stable regions.
    ascending_regions (list): List of tuples with start and end indices of ascending regions.
    descending_regions (list): List of tuples with start and end indices of descending regions.
    file_name_float (str): File path for the Excel file with timestamps in float.
    file_name_datetime (str): File path for the Excel file with timestamps in datetime format.
    """
    try:
        print(Fore.YELLOW + "Saving intervals to Excel files...")
        intervals_float = []
        intervals_datetime = []
        
        # Convert the timestamps to float (seconds) and keep as datetime
        timestamps_as_float = [timestamp.timestamp() for timestamp in timestamps]
        timestamps_as_datetime = [timestamp for timestamp in timestamps]

        # Function to process regions
        def process_regions(regions, region_type):
            for start, end in regions:
                # Ensure end index is within bounds
                if end >= len(timestamps_as_float):
                    end = len(timestamps_as_float) - 1

                start_time_float = timestamps_as_float[start]
                end_time_float = timestamps_as_float[end]
                elapsed_time = end_time_float - start_time_float

                start_time_datetime = timestamps_as_datetime[start]
                end_time_datetime = timestamps_as_datetime[end]

                # Add to the float version list
                intervals_float.append({
                    "Type": region_type,
                    "Begin": start_time_float,
                    "End": end_time_float,
                    "Elapsed": elapsed_time
                })

                # Add to the datetime version list
                intervals_datetime.append({
                    "Type": region_type,
                    "Begin": start_time_datetime,
                    "End": end_time_datetime,
                    "Elapsed": elapsed_time
                })

        # Process all regions
        process_regions(stable_regions, "Stable")
        process_regions(ascending_regions, "Ascending")
        process_regions(descending_regions, "Descending")

        # Convert the intervals lists to pandas DataFrames
        df_intervals_float = pd.DataFrame(intervals_float)
        df_intervals_datetime = pd.DataFrame(intervals_datetime)

        # Sort the DataFrames by the 'Begin' column to ensure chronological order
        df_intervals_float = df_intervals_float.sort_values(by="Begin")
        df_intervals_datetime = df_intervals_datetime.sort_values(by="Begin")

        # Format the datetime columns
        df_intervals_datetime['Begin'] = df_intervals_datetime['Begin'].dt.strftime('%Y-%m-%d %H:%M:%S:%f')
        df_intervals_datetime['End'] = df_intervals_datetime['End'].dt.strftime('%Y-%m-%d %H:%M:%S:%f')

        # Save both DataFrames to Excel files
        df_intervals_float.to_excel(file_name_float, index=False)
        df_intervals_datetime.to_excel(file_name_datetime, index=False)
        
        print(Fore.GREEN + f"Intervals saved to {file_name_float} (float seconds) and {file_name_datetime} (datetime format)")
    except Exception as e:
        print(Fore.RED + "Error in save_intervals_to_excel_chronologically function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise


# creating a function to make the data coming from different axis of the same lenght
def make_same_length(data):
    """
    Make the data coming from different axis of the same length.
    Args:
    data (pd.DataFrame): Dataframe with the data coming from different axis.
    Returns:
    pd.DataFrame: Dataframe with the data coming from different axis of the same length.
    """
    try:
        print(Fore.YELLOW + "Making data from different axis of the same length...")
        # Find the minimum length among the columns
        min_length = min(data.apply(len))

        # Truncate the data to the minimum length
        data_truncated = data.apply(lambda x: x[:min_length])

        print(Fore.GREEN + "Data from different axis made of the same length successfully.")
        return data_truncated
    except Exception as e:
        print(Fore.RED + "Error in make_same_length function:")
        print(Fore.RED + str(e))
        traceback.print_exc

def normalize_signal(signal):
    """
    Normalize the signal to have values between 0 and 1.

    Args:
    signal (np.array): Input signal.

    Returns:
    np.array: Normalized signal.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val)

def plot_normalized_signals(data, filtered_data, timestamps, value_columns):
    """
    Plot the normalized values of the three signals one above another.

    Args:
    data (pd.DataFrame): Original dataframe containing the signals.
    filtered_data (dict): Dictionary containing the filtered signals.
    timestamps (pd.DatetimeIndex): Timestamps of the signal.
    value_columns (list): List of column names representing the three signal values.
    """
    try:
        print(Fore.YELLOW + "Plotting normalized signals one above another...")

        # Create the subplots (one above another)
        fig, axes = plt.subplots(len(value_columns), 1, figsize=(14, 10), sharex=True)

        # Loop over each signal column
        for i, value_column in enumerate(value_columns):
            # Normalize original and filtered signals
            normalized_original = normalize_signal(data[value_column].values)
            normalized_filtered = normalize_signal(filtered_data[value_column])

            # Plot original and filtered signals
            axes[i].plot(timestamps, normalized_original, label=f'Original {value_column}', alpha=0.7)
            axes[i].plot(timestamps, normalized_filtered, label=f'Filtered {value_column}', alpha=0.7)

            # Set title and labels
            axes[i].set_title(f'Normalized {value_column}')
            axes[i].set_ylabel('Normalized Value')
            axes[i].legend()
            axes[i].grid(True)

        # Set common x-label
        axes[-1].set_xlabel('Time')

        # Tight layout for better spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()
        
        print(Fore.GREEN + "Plotting completed successfully.")

    except Exception as e:
        print(Fore.RED + "Error in plot_normalized_signals function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise
    
def plot_signal_all_together_plotly(data, timestamps):
    """
    Plot all the signals together using Plotly.

    Args:
    data (pd.DataFrame): Dataframe containing the signals.
    timestamps (pd.DatetimeIndex): Timestamps of the signal.
    """
    try:
        print(Fore.YELLOW + "Plotting all signals together using Plotly...")
        fig = go.Figure()

        # Loop over each signal column excluding the timestamp column
        for column in data.columns:
            if column != 'timestamp':
                fig.add_trace(go.Scatter
                (
                    x=timestamps,
                    y=data[column],
                    mode='lines',
                    name=column
                ))

        # Set the title and axis labels
        fig.update_layout(title='All Signals', xaxis_title='Time', yaxis_title='Value')

        # Show the plot
        fig.show()
        print(Fore.GREEN + "Plotting completed successfully.")

    except Exception as e:
        print(Fore.RED + "Error in plot_signal_all_together_plotly function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

import numpy as np
import plotly.graph_objs as go
from scipy.spatial.transform import Rotation as R
import pandas as pd
from colorama import Fore, Style, init
from tqdm import tqdm  # Progress bar

# Initialize colorama for colored debug prints
init(autoreset=True)

def rotation_matrix_from_euler_angles(euler_angles):
    roll, pitch, yaw = euler_angles
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return rotation.as_matrix()

import plotly.subplots as sp

def create_rotating_box_with_signals(data, timestamps, window_size=1000, downsample_factor=50, save_html=True):
    try:
        print("Starting to create a 3D animation of the rotating box with signals and tracking lines...")

        # Apply moving average filter to each Euler angle (value1, value2, value3)
        filtered_data = pd.DataFrame({
            'value1': moving_average(data['value1'].values, window_size),
            'value2': moving_average(data['value2'].values, window_size),
            'value3': moving_average(data['value3'].values, window_size)
        })

        # Downsample the filtered and raw data for faster processing
        data_downsampled = data.iloc[::downsample_factor, :].reset_index(drop=True)
        filtered_data_downsampled = filtered_data.iloc[::downsample_factor, :].reset_index(drop=True)
        timestamps_downsampled = timestamps[::downsample_factor]
        total_frames = len(data_downsampled)

        print(f"Data downsampled by a factor of {downsample_factor}. Total frames to generate: {total_frames}")

        # Ensure timestamps are in seconds (if they are datetime, convert to seconds)
        if isinstance(timestamps_downsampled[0], pd.Timestamp):
            timestamps_downsampled = (timestamps_downsampled - timestamps_downsampled[0]).total_seconds()
        else:
            timestamps_downsampled = np.array(timestamps_downsampled) - timestamps_downsampled[0]

        # Calculate total duration and average frame duration
        total_duration = timestamps_downsampled[-1] - timestamps_downsampled[0]
        average_frame_duration = total_duration / (total_frames - 1)
        average_frame_duration_ms = average_frame_duration * 1000  # Convert to milliseconds

        print(f"Total animation duration: {total_duration} seconds")
        print(f"Average frame duration: {average_frame_duration_ms} milliseconds")
        
        # Calculate min and max y-values for the signal plots
        min_y = min(data_downsampled[['value1', 'value2', 'value3']].min().min(),
                    filtered_data_downsampled[['value1', 'value2', 'value3']].min().min())
        max_y = max(data_downsampled[['value1', 'value2', 'value3']].max().max(),
                    filtered_data_downsampled[['value1', 'value2', 'value3']].max().max())

        # Create a subplot: Left side for the 3D box, Right side for the signals
        fig = sp.make_subplots(
            rows=1, cols=2, 
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],  # 3D plot on the left, XY plot on the right
            column_widths=[0.6, 0.4],
            subplot_titles=("3D Rotating Object", "Raw and Filtered Signals (Roll, Pitch, Yaw)")
        )

        # Initial signal traces on the right-hand side (raw and filtered signals)
        fig.add_trace(go.Scatter(x=timestamps_downsampled, y=data_downsampled['value1'],
                                 mode='lines', name='Raw Roll', line=dict(color='blue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=timestamps_downsampled, y=filtered_data_downsampled['value1'],
                                 mode='lines', name='Filtered Roll', line=dict(dash='dash', color='red')), row=1, col=2)

        fig.add_trace(go.Scatter(x=timestamps_downsampled, y=data_downsampled['value2'],
                                 mode='lines', name='Raw Pitch', line=dict(color='green')), row=1, col=2)
        fig.add_trace(go.Scatter(x=timestamps_downsampled, y=filtered_data_downsampled['value2'],
                                 mode='lines', name='Filtered Pitch', line=dict(dash='dash', color='purple')), row=1, col=2)

        fig.add_trace(go.Scatter(x=timestamps_downsampled, y=data_downsampled['value3'],
                                 mode='lines', name='Raw Yaw', line=dict(color='orange')), row=1, col=2)
        fig.add_trace(go.Scatter(x=timestamps_downsampled, y=filtered_data_downsampled['value3'],
                                 mode='lines', name='Filtered Yaw', line=dict(dash='dash', color='cyan')), row=1, col=2)

        # Create the initial vertical line
        vertical_line = go.Scatter(
            x=[timestamps_downsampled[0], timestamps_downsampled[0]],
            y=[min_y, max_y],
            mode='lines',
            line=dict(color='black', width=2),
            name='Current Frame',
            showlegend=False
        )
        fig.add_trace(vertical_line, row=1, col=2)

        # Create vertices of a 3D box (representing the phone)
        box_vertices = np.array([
            [-0.5, -1, -0.1], [0.5, -1, -0.1], [0.5, 1, -0.1], [-0.5, 1, -0.1],  # Bottom face
            [-0.5, -1, 0.1], [0.5, -1, 0.1], [0.5, 1, 0.1], [-0.5, 1, 0.1]      # Top face
        ])

        # Create the initial mesh of the box for the first frame using filtered data
        roll, pitch, yaw = np.radians([
            filtered_data_downsampled.iloc[0]['value1'],
            filtered_data_downsampled.iloc[0]['value2'],
            filtered_data_downsampled.iloc[0]['value3']
        ])
        rotation_matrix = rotation_matrix_from_euler_angles((roll, pitch, yaw))
        rotated_vertices = np.dot(box_vertices, rotation_matrix.T)
        box_trace = go.Mesh3d(
            x=rotated_vertices[:, 0],
            y=rotated_vertices[:, 1],
            z=rotated_vertices[:, 2],
            opacity=0.6,
            color="lightblue"
        )
        fig.add_trace(box_trace, row=1, col=1)

        # Set the y-axis range for the signal plot
        fig.update_yaxes(range=[min_y, max_y], row=1, col=2)

        # Set the scene aspect ratio to prevent rescaling
        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                zaxis=dict(range=[-1, 1])
            )
        )

        # Set up frames for animation using filtered Euler angles
        frames = []
        for i in tqdm(range(total_frames), desc="Generating frames", unit="frame"):
            roll, pitch, yaw = np.radians([
                filtered_data_downsampled.iloc[i]['value1'],
                filtered_data_downsampled.iloc[i]['value2'],
                filtered_data_downsampled.iloc[i]['value3']
            ])
            rotation_matrix = rotation_matrix_from_euler_angles((roll, pitch, yaw))

            # Rotate the box vertices by applying the rotation matrix
            rotated_vertices = np.dot(box_vertices, rotation_matrix.T)

            # Updated box trace
            box_trace_updated = go.Mesh3d(
                x=rotated_vertices[:, 0],
                y=rotated_vertices[:, 1],
                z=rotated_vertices[:, 2],
                opacity=0.6,
                color="lightblue"
            )

            # Updated vertical line
            vertical_line_updated = go.Scatter(
                x=[timestamps_downsampled[i], timestamps_downsampled[i]],
                y=[min_y, max_y],
                mode='lines',
                line=dict(color='black', width=2),
                name='Current Frame',
                showlegend=False
            )

            # Add frame with updated box and vertical line
            frames.append(go.Frame(
                data=[box_trace_updated, vertical_line_updated],
                traces=[7, 6],  # Indices of traces to update
                layout=go.Layout(
                    annotations=[
                        dict(x=0.5, y=1.05, xref='paper', yref='paper', showarrow=False,
                             text=(
                                 f"Roll: {filtered_data_downsampled.iloc[i]['value1']:.2f}°, "
                                 f"Pitch: {filtered_data_downsampled.iloc[i]['value2']:.2f}°, "
                                 f"Yaw: {filtered_data_downsampled.iloc[i]['value3']:.2f}°"
                             ),
                             font=dict(size=14, color="black"))
                    ]
                ),
                name=str(i)
            ))

        fig.frames = frames
        print("All frames generated successfully.")

        # Update the animation settings
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {"duration": average_frame_duration_ms, "redraw": True},
                            "fromcurrent": True
                        }],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
                        ],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [
                            [str(i)],
                            {"frame": {"duration": average_frame_duration_ms, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
                        ],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(total_frames)
                ],
                "active": 0
            }]
        )

        print("Animation setup completed successfully with accurate timing.")

        # Optionally save the animation as HTML
        if save_html:
            fig.write_html("rotation_animation_with_signals_and_timing.html")
            print("Animation saved as rotation_animation_with_signals_and_timing.html")

        # Show the plot
        fig.show()

    except Exception as e:
        print("Error in create_rotating_box_with_signals function:")
        print(str(e))
        traceback.print_exc()




# Example Usage (Make sure to load your data appropriately)
data = pd.DataFrame({
    'value1': np.random.rand(1000),
    'value2': np.random.rand(1000),
    'value3': np.random.rand(1000)
})
timestamps = pd.date_range(start='2024-01-01', periods=1000, freq='S')

def main():
    try:
        print(Fore.YELLOW + "Starting main function...")
        # Load the data
        file_path = config.file_path  # Use file_path from the configuration

        # Extract the athlete name and surname from the file path
        athlete = file_path.split('/')[-2] if len(file_path.split('/')) > 1 else "unknown_athlete"
        athlete_parts = athlete.split('_')
        athlete_name = "_".join(athlete_parts[:2]) if len(athlete_parts) >= 2 else "unknown_athlete"

        OUTPUT_PATH = os.path.join(config.output_path, athlete_name)

        # Append date of the file
        date = file_path.split('/')[-1].split('_')[1].split('.')[0] if '_' in file_path.split('/')[-1] else "unknown_date"
        OUTPUT_PATH = os.path.join(OUTPUT_PATH, date)

        # If path does not exist, create it
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        print(Fore.GREEN + f"Loading data from {file_path}")
        data = pd.read_csv(file_path, header=None, names=['timestamp', 'value1', 'value2', 'value3'])

        
        # Convert timestamps to datetime format
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')

        # Set the datetime as the index
        data.set_index('datetime', inplace=True)
        
        # make the data from different axis of the same length
        data = make_same_length(data)
        
        # Clean the dataframe
        data = clean_dataframe(data)
        
        filtered_data = {}
        # Iterate over each value column
        for value_column in ['value1', 'value2', 'value3']:
            # Apply the moving average filter with window size 250
            window_size = 250
            filtered_data[value_column] = moving_average(data[value_column].values, window_size)

        # Plot all signals together using Plotly
        #plot_signal_all_together_plotly(data, data.index)
        
        # Create an interactive 3D animation of the rotations using Plotly
        # create_rotating_box_with_signals(data, data.index)
        
        # create_animation_with_2d_plots_and_rotating_box (data, data.index)
        
        # Iterate over each value column
        for value_column in ['value1', 'value2', 'value3']:
            print(Fore.YELLOW + f"Processing {value_column}...")

            # Apply the moving average filter with window size from config
            window_size = config.window_size
            filtered_signal = moving_average(data[value_column].values, window_size)
            
            # Calculate the standard deviation of the filtered signal
            signal_std = np.nanstd(filtered_signal)
            print(Fore.GREEN + f"Standard deviation of filtered signal for {value_column}: {signal_std}")

            # Define the threshold for stable region detection using config
            threshold = config.variance_threshold * signal_std
            slope_threshold = config.slope_threshold
            print(Fore.GREEN + f"Variance threshold for stability for {value_column}: {threshold}")

            # Define the minimum length for stable regions using config
            min_length = config.min_length

            # Find stable regions
            stable_regions = find_stable_regions(filtered_signal, window_size, threshold, min_length)
            print(Fore.GREEN + f"Stable regions found for {value_column}: {stable_regions}")

            print(Fore.GREEN + "Going to find unstable regions...")
            unstable_regions = get_unstable_regions(filtered_signal, stable_regions)
            print(Fore.GREEN + f"Unstable regions found for {value_column}: {unstable_regions}")

            # Determine if unstable regions are ascending or descending
            ascending_regions, descending_regions = determine_if_unstable_regions_are_ascending_or_descending(filtered_signal, unstable_regions)
            print(Fore.GREEN + f"Ascending regions found for {value_column}: {ascending_regions}")
            print(Fore.GREEN + f"Descending regions found for {value_column}: {descending_regions}")

            # Plot the results using parameters from config
            plot_signal_and_regions_stable_ascending_descending_with_n_labels_on_x(
                data[value_column], filtered_signal, data.index, stable_regions, ascending_regions, descending_regions, 
                config.n_labels, output_path=OUTPUT_PATH
            )
            plot_signal_and_regions_stable_ascending_descending_with_n_labels_on_x_plotly(
                data[value_column], filtered_signal, data.index, stable_regions, ascending_regions, descending_regions, 
                config.n_labels, output_path=OUTPUT_PATH
            )

            # Save the regions to an Excel file in chronological order
            save_intervals_to_excel_chronologically(
                data.index, stable_regions, ascending_regions, descending_regions, 
                f"{OUTPUT_PATH}/signal_intervals_chronological_{value_column}.xlsx", 
                f"{OUTPUT_PATH}/signal_intervals_chronological_datetime_{value_column}.xlsx"
            )

            # Calculate the time of the stable regions
            stable_regions_time = []
            for start, end in stable_regions:
                if end >= len(data.index):  # Check to prevent out-of-bounds error
                    end = len(data.index) - 1
                start_time = data.index[start]
                end_time = data.index[end]
                stable_regions_time.append((start_time, end_time))


            # Save the regions to an Excel file in chronological order
            save_intervals_to_excel_chronologically(data.index, stable_regions, ascending_regions, descending_regions, f"signal_intervals_chronological_{value_column}.xlsx", OUTPUT_PATH + f"signal_intervals_chronological_datetime_{value_column}.xlsx")
            
            # Plot the normalized signals


        
        print(Fore.GREEN + "Main function completed successfully.")
    except Exception as e:
        print(Fore.RED + "An error occurred in the main function:")
        print(Fore.RED + str(e))
        traceback.print_exc()

if __name__ == "__main__":
    main()



# Example usage:
# df = pd.read_csv('path_to_file.csv')
# df = clean_dataframe(df)
