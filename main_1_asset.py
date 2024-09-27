import logging
from typing import Dict
from src.backtest_engine import BacktestEngine
from src.portfolio import Portfolio
from src.trade_manager import TradeManager
from src.strategy.depeg_strategy import DepegStrategy
from src.signal_database import SignalDatabase
from src.metrics import MetricsModule
import os

def setup_logging() -> logging.Logger:
    """
    Setup logging configuration for the application.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

def verify_files_exist(assets: Dict[str, str], logger: logging.Logger) -> bool:
    """
    Verify if the CSV files for each asset exist at the specified paths.

    Args:
        assets (Dict[str, str]): A dictionary with asset names as keys and their file paths as values.
        logger (logging.Logger): Logger instance for logging errors.

    Returns:
        bool: True if all files exist, False otherwise.
    """
    for asset_name, file_path in assets.items():
        if not os.path.exists(file_path):
            logger.error(f"CSV file for {asset_name} does not exist at path: {file_path}")
            return False
    logger.info("All asset files verified successfully.")
    return True

def main():
    """
    Main function to run the backtesting process.

    This function sets up the environment, initializes necessary components,
    and executes the backtest. It also handles error logging and signal analysis.
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting the backtesting process.")

    # Define the assets and their corresponding CSV data paths
    assets = {
        'EUTEUR': 'D:\\StableTrade_dataset\\EUTEUR_1m\\EUTEUR_1m_final_merged.csv'
    }

    # Verify if all required files exist before proceeding
    if not verify_files_exist(assets, logger):
        logger.error("File verification failed. Exiting the process.")
        return

    # Initialize components
    trade_manager = TradeManager()
    signal_database = SignalDatabase()
    logger.info("Trade manager and signal database initialized.")

    # Initialize strategies for the asset
    strategies = {
        'EUTEUR': DepegStrategy(
            market='EUTEUR',
            trade_manager=trade_manager,
            depeg_threshold=8,
            trade_amount=0.1,  # 10% of portfolio cash available
            stop_loss=0.02,  # 2% stop loss
            take_profit=0.05,  # 5% take profit
            trailing_stop=0.01  # 1% trailing stop
        )
    }
    logger.info(f"Strategies initialized for {', '.join(strategies.keys())}.")

    # Generate portfolio config
    portfolio_config = {
        asset_name: {
            'market_type': strategy.config['market_type'],
            'fees': strategy.config['fees'],
            'max_trades': strategy.config['max_trades'],
            'max_exposure': strategy.config['max_exposure']
        } for asset_name, strategy in strategies.items()
    }

    # Initialize portfolio
    initial_cash = 100000  # Example initial cash
    base_currency = 'EUR'
    portfolio = Portfolio(
        initial_cash=initial_cash,
        portfolio_config=portfolio_config,
        signal_database=signal_database,
        base_currency=base_currency
    )
    logger.info(f"Portfolio initialized with {initial_cash} {base_currency}.")

    # Initialize MetricsModule
    metrics_module = MetricsModule(base_currency=base_currency)
    logger.info("Metrics module initialized.")

    # Initialize the backtest engine
    backtest_engine = BacktestEngine(
        assets=assets,
        strategies=strategies,
        portfolio=portfolio,
        trade_manager=trade_manager,
        base_currency=base_currency,
        slippage=0.0001,  
        metrics=metrics_module,
        signal_database=signal_database
    )
    logger.info("Backtest engine initialized.")

    try:
        # Preprocess the data
        logger.info("Starting data preprocessing...")
        backtest_engine.preprocess_data()
        logger.info("Data preprocessing completed successfully.")

        # Run the backtest
        logger.info("Starting backtest execution...")
        backtest_engine.run_backtest()
        logger.info("Backtest completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the backtest: {e}", exc_info=True)
    
    logger.info("Backtesting process completed.")

if __name__ == "__main__":
    main()