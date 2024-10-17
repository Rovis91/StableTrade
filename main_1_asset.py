import os
import logging
from typing import Dict, Optional
from src.backtest_engine import BacktestEngine
from src.portfolio import Portfolio
from src.trade_manager import TradeManager
from src.strategy.depeg_strategy import DepegStrategy
from src.signal_database import SignalDatabase
from src.metrics import MetricsModule
from src.logger import setup_logger, set_log_levels


# Constants for initial configurations
INITIAL_CASH = 100000  # Example initial cash
BASE_CURRENCY = 'EUR'


def verify_files_exist(assets: Dict[str, str], logger: logging.Logger) -> bool:
    """
    Verify if the CSV files for each asset exist at the specified paths.

    Args:
        assets (Dict[str, str]): A dictionary with asset names as keys and their file paths as values.
        logger (logging.Logger): Logger instance for logging errors.

    Returns:
        bool: True if all files exist, False otherwise.

    Example:
        verify_files_exist({'asset1': 'path/to/file.csv'}, logger)
    """
    all_files_exist = True
    for asset_name, file_path in assets.items():
        if not os.path.exists(file_path):
            logger.error(f"CSV file for {asset_name} does not exist at path: {file_path}")
            all_files_exist = False

    if all_files_exist:
        logger.info("All asset files verified successfully.")
    
    return all_files_exist


def main(log_levels: Optional[Dict[str, str]] = None):
    """
    Main function to run the backtesting process.

    This function sets up the environment, initializes necessary components with specified log levels,
    and executes the backtest. It also handles error logging and signal analysis.

    Args:
        log_levels (Optional[Dict[str, str]]): A dictionary of component names and their log levels.
                                               Defaults to None, which means all components use the default log level.
    
    Example:
        main({'main': 'INFO', 'backtest_engine': 'DEBUG'})
    """
    if log_levels is None:
        log_levels = {}

    # Set log levels for all components
    set_log_levels(log_levels)

    # Setup main logging
    main_logger = setup_logger('main')
    main_logger.warning("Starting the backtesting process.")

    # Define the assets and their corresponding CSV data paths
    assets = {
        'EUTEUR': 'D:\\StableTrade_dataset\\EUTEUR_1m_2\\EUTEUR_1m_final_merged.csv'
    }

    # Verify if all required files exist before proceeding
    if not verify_files_exist(assets, main_logger):
        main_logger.error("File verification failed. Exiting the process.")
        return

    # Initialize TradeManager and SignalDatabase
    trade_manager = TradeManager(base_currency=BASE_CURRENCY)
    signal_database = SignalDatabase()
    main_logger.warning("Trade manager and signal database initialized.")

    # Initialize strategies for the asset
    strategies = {
        'EUTEUR': DepegStrategy(
            market='EUTEUR',
            trade_manager=trade_manager,
            depeg_threshold=8,
            trade_amount=0.1,
            stop_loss=0.02,
            take_profit=0.05,
            trailing_stop=0.01
        )
    }
    main_logger.warning(f"Strategies initialized for {', '.join(strategies.keys())}.")

    # Generate portfolio config based on strategies
    portfolio_config = {
        asset_name: {
            'market_type': strategy.config['market_type'],
            'fees': strategy.config['fees'],
            'max_trades': strategy.config['max_trades'],
            'max_exposure': strategy.config['max_exposure']
        } for asset_name, strategy in strategies.items()
    }

    # Initialize portfolio
    portfolio = Portfolio(
        initial_cash=INITIAL_CASH,
        portfolio_config=portfolio_config,
        signal_database=signal_database,
        trade_manager=trade_manager,
        base_currency=BASE_CURRENCY
    )
    main_logger.warning(f"Portfolio initialized with {INITIAL_CASH} {BASE_CURRENCY}.")

    # Initialize MetricsModule
    metrics_module = MetricsModule(portfolio=portfolio, base_currency=BASE_CURRENCY)
    main_logger.warning("Metrics module initialized.")

    # Initialize the backtest engine
    backtest_engine = BacktestEngine(
        assets=assets,
        strategies=strategies,
        portfolio=portfolio,
        trade_manager=trade_manager,
        base_currency=BASE_CURRENCY,
        slippage=0.0001,  
        metrics=metrics_module,
        signal_database=signal_database
    )
    main_logger.warning("Backtest engine initialized.")

    try:
        # Preprocess the data
        main_logger.warning("Starting data preprocessing...")
        backtest_engine.preprocess_data()
        main_logger.warning("Data preprocessing completed successfully.")

        # Run the backtest
        main_logger.warning("Starting backtest execution...")
        backtest_engine.run_backtest()
        main_logger.warning("Backtest completed successfully.")

    except Exception as e:
        main_logger.error(f"An error occurred during the backtest: {e}", exc_info=True)
    
    main_logger.warning("Backtesting process completed.")


if __name__ == "__main__":
    # Example of setting different log levels for components
    custom_log_levels = {
        'main': 'WARNING',
        'trade_manager': 'WARNING',
        'signal_database': 'WARNING',
        'depeg_strategy': 'WARNING',
        'portfolio': 'WARNING',
        'metrics': 'WARNING',
        'backtest_engine': 'WARNING'
    }
    main(custom_log_levels)
