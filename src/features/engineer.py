import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load turbofan engine data from text file and assign column names.
    
    Args:
        filepath: Path to the data file (train_FD001.txt, test_FD001.txt, etc.)
    
    Returns:
        DataFrame with proper column names
    """
    # Define column names for the 26 columns
    columns = ['unit', 'cycle', 
               'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    # Read space-separated file without headers
    data = pd.read_csv(filepath, sep='\s+', header=None, names=columns)
    
    return data


def calculate_rul(data):
    """
    Calculate Remaining Useful Life for each data point.
    Used for training data where engines run to failure.
    
    Args:
        data: DataFrame with 'unit' and 'cycle' columns
    
    Returns:
        DataFrame with added 'RUL' column
    """
    # Find the maximum cycle (failure point) for each engine
    max_cycles = data.groupby('unit')['cycle'].max()
    
    # Create a mapping of unit to max_cycle
    data = data.copy()
    data['max_cycle'] = data['unit'].map(max_cycles)
    
    # RUL = cycles remaining until failure
    data['RUL'] = data['max_cycle'] - data['cycle']
    
    # Drop temporary column
    data = data.drop('max_cycle', axis=1)
    
    return data


def load_test_rul(filepath):
    """
    Load true RUL values for test data.
    
    Args:
        filepath: Path to RUL file (RUL_FD001.txt)
    
    Returns:
        DataFrame with unit and RUL columns
    """
    rul_data = pd.read_csv(filepath, header=None, names=['RUL'])
    # Add unit numbers (1, 2, 3, ...)
    rul_data['unit'] = range(1, len(rul_data) + 1)
    
    return rul_data[['unit', 'RUL']]


def add_test_rul(test_data, rul_data):
    """
    Add RUL values to test dataset.
    Test data ends before failure, so we use provided RUL values.
    
    Args:
        test_data: Test DataFrame
        rul_data: DataFrame with true RUL values
    
    Returns:
        Test data with RUL column
    """
    # Get last cycle for each engine in test data
    last_cycles = test_data.groupby('unit')['cycle'].max().reset_index()
    last_cycles.columns = ['unit', 'last_cycle']
    
    # Merge test data with last cycle info
    test_data = test_data.merge(last_cycles, on='unit')
    
    # Merge with true RUL values
    test_data = test_data.merge(rul_data, on='unit')
    
    # Calculate RUL for each row
    # RUL at any cycle = true_RUL + (last_cycle - current_cycle)
    test_data['RUL'] = test_data['RUL'] + (test_data['last_cycle'] - test_data['cycle'])
    
    # Drop temporary column
    test_data = test_data.drop('last_cycle', axis=1)
    
    return test_data


def add_rolling_features(data, window_sizes=[5, 10, 20]):
    """
    Create rolling statistics features to capture trends.
    
    Args:
        data: DataFrame with sensor columns
        window_sizes: List of window sizes for rolling calculations
    
    Returns:
        DataFrame with added rolling features
    """
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    
    data = data.copy()
    
    # For each sensor and window size, calculate rolling mean and std
    for sensor in sensor_cols:
        for window in window_sizes:
            # Rolling mean
            data[f'{sensor}_rolling_mean_{window}'] = data.groupby('unit')[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling standard deviation
            data[f'{sensor}_rolling_std_{window}'] = data.groupby('unit')[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
    
    # Fill NaN values with 0 (for initial rows where window isn't full)
    data = data.fillna(0)
    
    return data


def add_lag_features(data, lags=[1, 5, 10]):
    """
    Add lagged sensor values as features (previous cycle values).
    
    Args:
        data: DataFrame with sensor columns
        lags: List of lag periods
    
    Returns:
        DataFrame with added lag features
    """
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    
    data = data.copy()
    
    # For each sensor and lag period
    for sensor in sensor_cols:
        for lag in lags:
            # Shift values within each engine group
            data[f'{sensor}_lag_{lag}'] = data.groupby('unit')[sensor].shift(lag)
    
    # Fill NaN values with 0 (for initial cycles without history)
    data = data.fillna(0)
    
    return data


def remove_constant_sensors(data):
    """
    Remove sensors that don't vary (provide no information).
    Based on standard analysis of NASA dataset.
    
    Args:
        data: DataFrame with sensor columns
    
    Returns:
        DataFrame with constant sensors removed
    """
    # Sensors that show no variation in FD001 dataset
    constant_sensors = ['sensor_1', 'sensor_5', 'sensor_10', 
                       'sensor_16', 'sensor_18', 'sensor_19']
    
    # Also remove op_setting_3 (constant in FD001)
    constant_cols = constant_sensors + ['op_setting_3']
    
    # Drop columns if they exist
    cols_to_drop = [col for col in constant_cols if col in data.columns]
    data = data.drop(columns=cols_to_drop)
    
    return data


def engineer_features(data, is_train=True, rul_filepath=None, 
                     add_rolling=True, add_lags=True):
    """
    Main feature engineering pipeline.
    """
    # Calculate or load RUL
    if is_train:
        data = calculate_rul(data)
    else:
        if rul_filepath is None:
            raise ValueError("rul_filepath required for test data")
        rul_data = load_test_rul(rul_filepath)
        data = add_test_rul(data, rul_data)
    
    # Add rolling features FIRST (before removing sensors)
    if add_rolling:
        data = add_rolling_features(data, window_sizes=[5, 10, 20])
    
    # Add lag features
    if add_lags:
        data = add_lag_features(data, lags=[1, 5, 10])
    
    # Remove constant sensors LAST
    data = remove_constant_sensors(data)
    
    return data


def save_processed_data(data, filepath):
    """
    Save processed data to CSV file.
    
    Args:
        data: Processed DataFrame
        filepath: Output file path
    """
    data.to_csv(filepath, index=False)
    print(f"Saved processed data to {filepath}")


if __name__ == "__main__":
    # Load and process training data
    train_data = load_data('data/raw/train_FD001.txt')
    train_processed = engineer_features(train_data, is_train=True)
    save_processed_data(train_processed, 'data/processed/train_FD001_processed.csv')
    
    # Load and process test data
    test_data = load_data('data/raw/test_FD001.txt')
    test_processed = engineer_features(test_data, is_train=False, 
                                      rul_filepath='data/raw/RUL_FD001.txt')
    save_processed_data(test_processed, 'data/processed/test_FD001_processed.csv')
    
    print("Feature engineering complete!")