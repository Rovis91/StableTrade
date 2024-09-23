import logging
from src.backtest_engine import BacktestEngine
from src.portfolio import Portfolio
from src.trade_manager import TradeManager
from src.strategy.depeg_strategy import DepegStrategy
import os

def setup_logging():
    """
    Setup logging configuration.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def verify_files_exist(assets, logger):
    """
    Verify if the files for each asset exist.

    Args:
        assets (dict): A dictionary with asset names and their file paths.
        logger: Logger instance for logging errors.

    Returns:
        bool: True if all files exist, False otherwise.
    """
    for asset_name, file_path in assets.items():
        if not os.path.exists(file_path):
            logger.error(f"CSV file for {asset_name} does not exist at path: {file_path}")
            return False
    return True

def main():
    # Setup logging
    logger = setup_logging()

    # Initialize portfolio and trade manager with fees for each asset
    initial_cash = 100000  # Example initial cash
    fees = {
        'EUTEUR': {'entry': 0.001, 'exit': 0.001}  # 0.1% entry and exit fees for EUTEUR
    }
    trade_manager = TradeManager()
    portfolio = Portfolio(initial_cash=initial_cash, trade_manager=trade_manager, fees=fees)

    # Define the asset and its corresponding CSV data paths (for preprocessing and preprocessed data)
    assets = {
        'EUTEUR': 'D:\\StableTrade_dataset\\EUTEUR_1m\\EUTEUR_1m_final_merged.csv'
    }

    # Verify if all required files exist before proceeding
    if not verify_files_exist(assets, logger):
        logger.error("File verification failed. Exiting the process.")
        return

    # Initialize strategies for the asset
    strategies = {
        'EUTEUR': DepegStrategy(
            market='EUTEUR', 
            trade_manager=trade_manager, 
            depeg_threshold=5, 
            trade_amount=0.1, 
            stop_loss=None, 
            take_profit=None, 
            trailing_stop=None
        )
    }

    # Initialize the backtest engine
    backtest_engine = BacktestEngine(
        assets=assets, 
        strategies=strategies, 
        portfolio=portfolio, 
        trade_manager=trade_manager
    )
    
    try:
        # Preprocess the data
        logger.info("Starting data preprocessing...")
        backtest_engine.preprocess_data()
        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}", exc_info=True)
        return  # Exit if preprocessing fails

    try:
        # Run the backtest
        logger.info("Starting backtest...")
        backtest_engine.run_backtest()
        logger.info("Backtest completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the backtest: {e}", exc_info=True)

if __name__ == "__main__":
    main()
