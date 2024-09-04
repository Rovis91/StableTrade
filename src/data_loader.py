import pandas as pd

def load_csv(file_path):
    """
    Loads the CSV file containing historical price data.
    
    :param file_path: Path to the CSV file
    :return: DataFrame with historical data (timestamp, open, high, low, close, volume)
    """
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    # Ensure it has necessary columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {', '.join(required_columns)}")
    
    return df
