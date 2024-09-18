import pandas as pd
import logging
import os

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
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self._validate_paths()

    def _validate_paths(self):
        """
        Validate input and output paths.
        """
        if not os.path.isfile(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if not os.access(os.path.dirname(self.output_path), os.W_OK):
            raise PermissionError(f"Cannot write to output path: {self.output_path}")

    def preprocess_data(self, required_indicators: dict):
        try:
            logging.info("Loading data for preprocessing...")
            data = pd.read_csv(self.input_path)
            
            # Validate data format once
            self._validate_data_format(data)

            data.set_index('timestamp', inplace=True)

            for indicator, params in required_indicators.items():
                if indicator in indicator_registry:
                    for param in params:
                        try:
                            indicator_registry[indicator](data, param)
                        except Exception as e:
                            logging.error(f"Error computing {indicator} with param {param}: {e}")
                else:
                    logging.warning(f"Indicator '{indicator}' is not implemented.")
            
            logging.info(f"Saving enriched data to {self.output_path}")
            data.to_csv(self.output_path)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
        except KeyError as e:
            logging.error(f"Missing required column: {e}")
        except Exception as e:
            logging.error(f"An error occurred during preprocessing: {e}")
            raise

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
    
    # We assume 'close' column has already been validated in preprocess_data
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
    
    # We assume 'close' column has already been validated in preprocess_data
    data[column_name] = data['close'].ewm(span=period, adjust=False).mean()
    logging.info(f"Computed {column_name}.")
