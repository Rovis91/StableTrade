import pandas as pd
from src.logger import setup_logger

# Create a logger for this module
logger = setup_logger(__name__)

class DataPreprocessor:
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the DataPreprocessor.

        Args:
            input_path (str): Path to the original data CSV file.
            output_path (str): Path to save the enriched data CSV file.
        """
        self.input_path = input_path
        self.output_path = output_path

    def preprocess_data(self, required_indicators: dict):
        try:
            logger.info("Loading data for preprocessing...")
            data = pd.read_csv(self.input_path)

            # Ensure 'timestamp' remains in milliseconds
            if 'timestamp' not in data.columns:
                logger.error("'timestamp' column is missing in the input data.")
                return

            data.set_index('timestamp', inplace=True)

            # Compute each required indicator
            for indicator, params in required_indicators.items():
                if hasattr(self, f"compute_{indicator.lower()}"):
                    for param in params:
                        # Check if the indicator is already computed
                        column_name = f"{indicator.upper()}_{param}"
                        if column_name in data.columns:
                            logger.warning(f"{column_name} already exists. Skipping computation.")
                            continue
                        getattr(self, f"compute_{indicator.lower()}")(data, param)
                else:
                    logger.warning(f"Indicator '{indicator}' is not implemented. Skipping.")

            # Save the enriched data to a new file without changing the timestamp format
            logger.info(f"Saving enriched data to {self.output_path}")
            data.to_csv(self.output_path)
        except Exception as e:
            logger.error(f"An error occurred during preprocessing: {e}")

    def compute_sma(self, data: pd.DataFrame, period: int):
        """
        Compute Simple Moving Average (SMA) and add it to the DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing market data.
            period (int): The period for the SMA calculation.
        """
        # Ensure the 'close' column exists
        if 'close' not in data.columns:
            logger.warning("Missing 'close' column. Cannot compute SMA.")
            return
        
        column_name = f"SMA_{period}"
        data[column_name] = data['close'].rolling(window=period).mean()
        logger.info(f"Computed {column_name}.")

    def compute_ema(self, data: pd.DataFrame, period: int):
        """
        Compute Exponential Moving Average (EMA) and add it to the DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing market data.
            period (int): The period for the EMA calculation.
        """
        # Ensure the 'close' column exists
        if 'close' not in data.columns:
            logger.warning("Missing 'close' column. Cannot compute EMA.")
            return

        column_name = f"EMA_{period}"
        data[column_name] = data['close'].ewm(span=period, adjust=False).mean()
        logger.info(f"Computed {column_name}.")
