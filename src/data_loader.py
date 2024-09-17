from src.logger import setup_logger
import pandas as pd

# Create a logger for this module
logger = setup_logger(__name__)

class DataLoader:
    def __init__(self, data_path: str, required_columns: list = None, timestamp_unit: str = 'ms'):
        """
        Initialize the DataLoader with the path to the data file.

        Args:
            data_path (str): Path to the CSV data file.
            required_columns (list, optional): List of required columns.
            timestamp_unit (str): Unit of the timestamp ('ms' for milliseconds).
        """
        self.data_path = data_path
        self.required_columns = required_columns or ['open', 'high', 'low', 'close', 'volume']
        self.timestamp_unit = timestamp_unit
        logger.info(f"DataLoader initialized with data path: {data_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load and validate the CSV data.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.

        Raises:
            ValueError: If the CSV is empty or missing required columns.
        """
        data = self._read_csv()
        self._validate_data(data)
        self._parse_timestamps(data)
        logger.info(f"Data loaded successfully with shape: {data.shape}")
        return data

    def _read_csv(self) -> pd.DataFrame:
        """Read the CSV file into a DataFrame."""
        try:
            logger.info("Reading CSV data...")
            return pd.read_csv(self.data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {self.data_path}")
            raise

    def _validate_data(self, data: pd.DataFrame):
        """Validate the DataFrame to ensure it contains required columns."""
        if data.empty:
            raise ValueError("CSV file is empty.")
        
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"CSV file is missing the following columns: {', '.join(missing_cols)}")

    def _parse_timestamps(self, data: pd.DataFrame):
        """Parse the timestamp column and set it as the DataFrame index."""
        try:
            logger.info("Parsing timestamps...")
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit=self.timestamp_unit)
            data.set_index('timestamp', inplace=True)
        except Exception as e:
            logger.error(f"Error parsing timestamps: {e}")
            raise
