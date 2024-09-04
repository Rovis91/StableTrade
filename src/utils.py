import os
from dotenv import load_dotenv
import logging
import json

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

def get_env_var(var_name, default=None):
    """
    Utility function to get an environment variable.
    """
    return os.getenv(var_name, default)


def log_backtest_results(result):
    """
    Log the results of a backtest.
    """
    logging.info(f"Backtest completed. Final balance: {result['final_balance']}, "
                 f"Sharpe Ratio: {result['sharpe_ratio']}, "
                 f"Trades: {result['number_of_trades']}")

def save_results_to_json(file_path, results):
    """
    Save backtest results to a JSON file.
    
    :param file_path: Path to the output JSON file
    :param results: Results of the backtest to be saved
    """
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
