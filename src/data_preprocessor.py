import os
import hashlib
import pandas as pd
import logging

# Global indicator registry dictionary
indicator_registry = {}

def register_indicator(name):
    """
    Decorator to register a new indicator in the global registry.
    """
    def decorator(indicator_func):
        if name in indicator_registry:
            logging.warning(f"Overwriting existing indicator: {name}")
        indicator_registry[name] = indicator_func
        return indicator_func
    return decorator

class DataPreprocessor:
    def __init__(self, input_path: str):
        """
        Initialize the DataPreprocessor.

        Args:
            input_path (str): The path to the raw input CSV file.
        """
        self.input_path = input_path
        self.output_dir = self._extract_directory(input_path)
        self._validate_paths()

    def _extract_directory(self, input_path: str) -> str:
        """
        Extract the directory path from the input file path.
        """
        return os.path.dirname(input_path)

    def _validate_paths(self):
        """
        Validate input path and output directory.
        """
        if not os.path.isfile(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if not os.access(self.output_dir, os.W_OK):
            raise PermissionError(f"Cannot write to output directory: {self.output_dir}")

    def _generate_output_filename(self, required_indicators: dict) -> str:
        """
        Generate a unique filename based on the required indicators.

        Args:
            required_indicators (dict): A dictionary of indicators and their parameters.

        Returns:
            str: A unique filename with a hash of the indicators appended.
        """
        indicators_str = str(required_indicators)
        indicators_hash = hashlib.md5(indicators_str.encode()).hexdigest()
        base_name = os.path.basename(self.input_path).replace('.csv', '')
        return f"{base_name}_indicators_{indicators_hash}.csv"

    def preprocess_data(self, required_indicators: dict):
        """
        Preprocess the data, adding the required indicators. If a preprocessed file with the correct
        indicators already exists, it will be loaded instead of recalculating.

        Args:
            required_indicators (dict): A dictionary of indicators and their parameters.
        """
        # Generate full output path based on indicators
        output_filename = self._generate_output_filename(required_indicators)
        preprocessed_file_path = os.path.join(self.output_dir, output_filename)

        if os.path.exists(preprocessed_file_path):
            logging.info(f"Loading preprocessed data from {preprocessed_file_path}")
            data = pd.read_csv(preprocessed_file_path, index_col='timestamp')
            return data

        # Load raw data and apply indicators if no preprocessed file exists
        logging.info("Loading data for preprocessing...")
        data = pd.read_csv(self.input_path)
        self._validate_data_format(data)
        data.set_index('timestamp', inplace=True)

        # Apply indicators
        for indicator, params in required_indicators.items():
            if indicator in indicator_registry:
                for param in params:
                    try:
                        indicator_registry[indicator](data, param)
                    except Exception as e:
                        logging.error(f"Error computing {indicator} with param {param}: {e}")
            else:
                logging.warning(f"Indicator '{indicator}' is not implemented.")

        # Save enriched data with indicators
        logging.info(f"Saving preprocessed data to {preprocessed_file_path}")
        data.to_csv(preprocessed_file_path)
        return data

    def _validate_data_format(self, data: pd.DataFrame):
        """
        Validate the data format to ensure required columns are present.

        Args:
            data (pd.DataFrame): The DataFrame containing market data.
        """
        required_columns = ['timestamp', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise KeyError(f"Input data is missing required columns: {', '.join(missing_columns)}")

@register_indicator('SMA')
def compute_sma(data: pd.DataFrame, period: int):
    """
    Compute Simple Moving Average (SMA) and add it to the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing market data.
        period (int): The period for the SMA calculation.
    """
    column_name = f"SMA_{period}"
    data[column_name] = data['close'].rolling(window=period).mean()
    logging.info(f"Computed {column_name}.")

@register_indicator('EMA')
def compute_ema(data: pd.DataFrame, period: int):
    """
    Compute Exponential Moving Average (EMA) and add it to the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing market data.
        period (int): The period for the EMA calculation.
    """
    column_name = f"EMA_{period}"
    data[column_name] = data['close'].ewm(span=period, adjust=False).mean()
    logging.info(f"Computed {column_name}.")
