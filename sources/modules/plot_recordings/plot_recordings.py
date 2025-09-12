import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore
import traceback

# Initialize colorama for colored debug prints
init(autoreset=True)

def load_config_from_yaml(yaml_path):
    """
    Loads the YAML configuration file.
    Args:
        yaml_path (str): Path to the YAML configuration file.
    Returns:
        dict: Configuration parameters.
    """
    import yaml
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(Fore.RED + "Error while loading YAML configuration:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        return {}

def moving_average(signal, window_size):
    """
    Apply a moving average filter to a signal.
    Args:
        signal (np.array): Input signal.
        window_size (int): Size of the moving average window.
    Returns:
        np.array: Filtered signal.
    """
    try:
        print(Fore.GREEN + f"Applying moving average filter with window size: {window_size}")
        result = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
        print(Fore.GREEN + "Moving average filter applied successfully.")
        return result
    except Exception as e:
        print(Fore.RED + "Error in moving_average function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def filter_signal(signal, window_size, threshold):
    """
    Remove spikes from a signal, replacing them with the median of the local window, 
    then apply a mean filter.
    Args:
        signal (np.array): Input signal.
        window_size (int): Size of the sliding window (must be odd).
        threshold (float): Threshold for spike detection.
    Returns:
        np.array: Processed (filtered) signal.
    """
    try:
        # Ensure window size is odd
        if window_size % 2 == 0:
            print(Fore.YELLOW + f"Window size {window_size} is even. Adjusting to {window_size + 1}.")
            window_size += 1

        # Step 1: Spike removal
        print(Fore.GREEN + f"Detecting and removing spikes with threshold: {threshold}")
        pad_size = window_size // 2
        padded_signal = np.pad(signal, (pad_size, pad_size), mode='reflect')
        
        filtered_signal = signal.copy()  # Create a copy to preserve structure
        for i in range(len(signal)):
            window = padded_signal[i:i + window_size]
            median = np.median(window)
            if abs(signal[i] - median) >= threshold:
                filtered_signal[i] = median

        print(Fore.GREEN + "Spike removal completed.")

        # Step 2: Mean filter
        print(Fore.GREEN + "Applying mean filter...")
        mean_filter = np.ones(window_size) / window_size
        smoothed_signal = np.convolve(filtered_signal, mean_filter, mode='same')
        print(Fore.GREEN + "Mean filter applied successfully.")

        return smoothed_signal
    except Exception as e:
        print(Fore.RED + "Error in filter_signal function:")
        print(Fore.RED + str(e))
        traceback.print_exc()
        raise

def check_numeric_header(df):
    """
    Check if the header of the dataframe is numeric.
    Args:
        df (pd.DataFrame): Input dataframe.
    Returns:
        bool: True if the header is numeric, False otherwise.
    """
    return all(isinstance(col, (int, float)) for col in df.columns)

def rename_columns(df, new_columns):
    """
    Rename the columns of the dataframe.
    Args:
        df (pd.DataFrame): Input dataframe.
        new_columns (list): List of new column names.
    """
    df.columns = new_columns

def plot_all_recordings(config):
    """
    Plot all recordings based on the configuration.
    Args:
        config (dict): Configuration parameters.
    """
    try:
        print(Fore.YELLOW + "Starting to plot all recordings...")

        # Set output path relative to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, config['output_path'])

        # Ensure output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(Fore.GREEN + f"Created output directory at {output_path}")

        # Read CSV file from config paths
        file_path = os.path.join(script_dir, config['input_path'], config['file_name'])
        print(Fore.GREEN + f"Loading data from {file_path}")

        data = pd.read_csv(file_path)

        # Check if the file has a header, if not, create one
        if data.columns[0] != 'timestamp':
            print(Fore.YELLOW + "No header found in CSV. Assuming order: timestamp, x, y, z")
            data.columns = ['timestamp', 'x', 'y', 'z']

        # Check if the header is numeric and rename columns accordingly
        if check_numeric_header(data):
            print(Fore.YELLOW + "Numeric header found. Renaming columns...")
            rename_columns(data, ['timestamp', 'x', 'y', 'z'])

        # For demonstration, assume 'timestamp' is present or build a numeric one
        if 'timestamp' not in data.columns:
            data['timestamp'] = np.arange(len(data))  # simple fallback

        # Convert timestamp to datetime if not already
        # Here, we assume data['timestamp'] in seconds or ms. Adjust as needed:
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')  
        data.set_index('datetime', inplace=True)

        # Optionally make the same length, if you had multiple columns
        min_length = min(data.apply(len))
        data = data.apply(lambda x: x[:min_length])

        # Prepare to filter the signals
        filtered_data = {}
        for col in data.columns:
            if col != 'timestamp':
                filtered_data[col] = filter_signal(data[col].values, config['filter_window_size'], config['spike_threshold'])

        # Plot each axis separately
        for axis in ['value1', 'value2', 'value3']:
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(data.index, data[axis], label=f'Original {axis}', alpha=0.7)
            ax.plot(data.index, filtered_data[axis], label=f'Filtered {axis}', alpha=0.7)
            ax.set_title(f'Recordings for {axis}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'{axis}_recordings.png'))
            plt.close()
            print(Fore.GREEN + f"Plotting for {axis} completed successfully.")

    except Exception as e:
        print(Fore.RED + "Error in plot_all_recordings function:")
        print(Fore.RED + str(e))
        traceback.print_exc()

if __name__ == "__main__":
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "plot_recordings.yaml")
    config = load_config_from_yaml(yaml_path)

    # Plot all recordings
    plot_all_recordings(config)
