import os
import json
import pandas as pd
import requests
import logging
import time
import signal
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class HistoricalDataFetcher(ABC):
    REQUIRED_FIELDS = ["platform", "pair", "start", "timeframe", "output_path", "liquid"]

    def __init__(self, config_path):
        start_time = time.time()
        self.config = self._load_config(config_path)
        self.output_dir = self._create_directory(self.config['output_path'], f"{self.config['pair']}_{self.config['timeframe']}")
        self.final_file = os.path.join(self.output_dir, f"{self.config['pair']}_{self.config['timeframe']}_final_merged.csv")
        self.last_timestamp = self._get_last_timestamp()
        self._register_signal_handlers()
        logging.info(f"Initialization completed in {time.time() - start_time:.2f} seconds")

    def _load_config(self, config_path):
        """Load and validate configuration."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            for field in self.REQUIRED_FIELDS:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            config['start'] = self._ensure_mts(self._convert_timestamp(config['start']))
            config['end'] = self._ensure_mts(self._convert_timestamp(config.get('end', 'now')))
            return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Error reading configuration file: {e}")
            raise

    @staticmethod
    def _convert_timestamp(timestamp):
        """Convert a timestamp to milliseconds since epoch."""
        if timestamp is None or timestamp == 'now':
            return int(datetime.now(timezone.utc).timestamp() * 1000)
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp)
                return int(dt.timestamp() * 1000)
            return int(timestamp)
        except (ValueError, TypeError) as e:
            logging.error(f"Invalid timestamp format: {timestamp}")
            raise ValueError("Invalid timestamp format") from e

    @staticmethod
    def _ensure_mts(timestamp):
        """Ensure the timestamp is in milliseconds since epoch (MTS)."""
        if len(str(timestamp)) == 13:
            return timestamp  # Already in MTS
        elif len(str(timestamp)) == 10:  # Assume it's in seconds
            return timestamp * 1000
        else:
            logging.error(f"Invalid timestamp format: {timestamp}")
            raise ValueError(f"Invalid timestamp format: {timestamp}")

    @staticmethod
    def _create_directory(base_path, sub_path):
        """Create the output directory if it doesn't exist."""
        output_dir = os.path.join(base_path, sub_path)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_graceful_shutdown)
        signal.signal(signal.SIGTERM, self._handle_graceful_shutdown)
        logging.info("Signal handlers for graceful shutdown registered.")

    def fetch_data(self):
        """Fetch data from the API."""
        retries = 0
        while self.last_timestamp < self.config['end']:
            try:
                data = self._fetch_from_api()
                if not data:
                    logging.info("No more data to fetch.")
                    break
                self._save_data(data)
                self.last_timestamp = data[-1][0] + 60000  # Move to the next timestamp
                time.sleep(self.sleep_time)
                retries = 0
            except requests.RequestException:
                retries += 1
                if retries > self.max_retries:
                    logging.error("Max retries exceeded. Exiting fetch loop.")
                    break
                logging.warning(f"Request failed. Retrying in {2 ** retries} seconds...")
                time.sleep(2 ** retries)

    def _fetch_from_api(self):
        """Fetch data from the API."""
        response = requests.get(self.get_url(self.last_timestamp), headers=self.authenticate())
        response.raise_for_status()
        data = response.json()
        self._validate_data(data)
        logging.info(f"Fetched {len(data)} records from API")
        return data

    def _save_data(self, data):
        """Save data directly to the final merged CSV."""
        df = pd.DataFrame(data, columns=self.get_expected_columns()).sort_values(by='timestamp')
        mode, header = ('a', False) if os.path.exists(self.final_file) else ('w', True)
        df.to_csv(self.final_file, mode=mode, header=header, index=False)
        logging.info(f"Appended {len(data)} records to {self.final_file}")

    def _get_last_timestamp(self):
        """Get the last timestamp from the most recent entry in the final merged CSV file."""
        try:
            if not os.path.exists(self.final_file):
                logging.info("No final merged file found. Using start timestamp from configuration.")
                return self.config['start']
            df = pd.read_csv(self.final_file)
            last_timestamp = df['timestamp'].iloc[-1] if not df.empty else self.config['start']
            logging.info(f"Last timestamp retrieved: {last_timestamp}")
            return last_timestamp
        except Exception:
            logging.warning("Error retrieving last timestamp. Defaulting to start timestamp.")
            return self.config['start']

    def _handle_graceful_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logging.info("Graceful shutdown initiated.")
        sys.exit(0)

    def _validate_data(self, data):
        """Validate the fetched data to ensure integrity."""
        if not all(isinstance(item, list) and len(item) == len(self.get_expected_columns()) for item in data):
            logging.error("Invalid data format received.")
            raise ValueError("Invalid data format")

    def _check_data_integrity(self):
        """Ensure that timestamps in the final merged file are in order."""
        if not os.path.exists(self.final_file):
            logging.warning("No final merged file found. Skipping integrity check.")
            return

        try:
            df = pd.read_csv(self.final_file)
            if df['timestamp'].is_monotonic_increasing:
                logging.info("Data integrity check passed: All timestamps are in order.")
            else:
                logging.warning("Data integrity check failed: Timestamps are not in order.")
        except Exception as e:
            logging.error(f"Error during data integrity check: {e}")

    @abstractmethod
    def get_url(self, start_timestamp):
        pass

    @abstractmethod
    def authenticate(self):
        pass

    @abstractmethod
    def get_expected_columns(self):
        pass

    def run(self):
        """Run the data fetching process."""
        logging.info("Starting data fetching process.")
        self.fetch_data()
        self._check_data_integrity()


class BitfinexFetcher(HistoricalDataFetcher):
    def __init__(self, config_path, rate_limit=25, max_retries=5, row_limit=1000):
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.row_limit = row_limit
        self.sleep_time = 60 / rate_limit
        super().__init__(config_path)

    def get_url(self, start_timestamp):
        base_url = "https://api-pub.bitfinex.com/v2/candles"
        return f"{base_url}/trade:{self.config['timeframe']}:t{self.config['pair']}/hist?sort=1&start={start_timestamp}&limit={self.row_limit}"

    def authenticate(self):
        return {"accept": "application/json"}

    def get_expected_columns(self):
        return ['timestamp', 'open', 'close', 'high', 'low', 'volume']


def main():
    config_path = os.getenv('CONFIG_PATH', './data/config.json')
    fetcher = BitfinexFetcher(config_path)
    fetcher.run()


if __name__ == "__main__":
    main()
