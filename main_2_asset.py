import logging
from src.backtest_engine import BacktestEngine
from src.portfolio import Portfolio
from src.trade_manager import TradeManager
from src.strategy.depeg_strategy import DepegStrategy
from src.signal_database import SignalDatabase
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

    # Define the assets and their corresponding CSV data paths
    assets = {
        'EUTEUR': 'D:\\StableTrade_dataset\\EUTEUR_1m\\EUTEUR_1m_final_merged.csv',
        'EURUST': 'D:\\StableTrade_dataset\\EURUST_1m\\EURUST_1m_final_merged.csv'
    }

    # Verify if all required files exist before proceeding
    if not verify_files_exist(assets, logger):
        logger.error("File verification failed. Exiting the process.")
        return

    # Initialize trade manager
    trade_manager = TradeManager()

    # Initialize signal database
    signal_database = SignalDatabase()

    # Initialize strategies for both assets
    strategies = {
        'EUTEUR': DepegStrategy(
            market='EUTEUR',
            trade_manager=trade_manager,
            depeg_threshold=5,
            trade_amount=0.1,  # 10% of portfolio cash available
            stop_loss=0.02,  # 2% stop loss
            take_profit=0.05,  # 5% take profit
            trailing_stop=0.01  # 1% trailing stop
        ),
        'EURUST': DepegStrategy(
            market='EURUST',
            trade_manager=trade_manager,
            depeg_threshold=5,
            trade_amount=0.1,
            stop_loss=0.02,
            take_profit=0.05,
            trailing_stop=0.01
        )
    }

    # Generate portfolio config
    portfolio_config = {}
    for asset_name, strategy in strategies.items():
        portfolio_config[asset_name] = {
            'market_type': strategy.config['market_type'],
            'fees': strategy.config['fees'],
            'max_trades': strategy.config['max_trades'],
            'max_exposure': strategy.config['max_exposure']
        }

    # Initialize portfolio
    initial_cash = 100000  # Example initial cash
    portfolio = Portfolio(
        initial_cash=initial_cash,
        portfolio_config=portfolio_config,
        base_currency='EUR'
    )

    # Initialize the backtest engine
    backtest_engine = BacktestEngine(
        assets=assets,
        strategies=strategies,
        portfolio=portfolio,
        trade_manager=trade_manager,
        signal_database=signal_database
    )

    try:
        # Preprocess the data
        logger.info("Starting data preprocessing...")
        backtest_engine.preprocess_data()
        logger.info("Data preprocessing completed successfully.")

        # Run the backtest
        logger.info("Starting backtest...")
        backtest_engine.run_backtest()
        logger.info("Backtest completed successfully.")

        # Export signals to CSV for analysis
        signal_database.export_signals_to_csv("backtest_signals.csv")

        # Print signal statistics
        signal_stats = signal_database.get_signal_stats()
        logger.info("Signal Statistics:")
        for stat, value in signal_stats.items():
            logger.info(f"{stat}: {value}")

    except Exception as e:
        logger.error(f"An error occurred during the backtest: {e}", exc_info=True)

if __name__ == "__main__":
    main()