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
INITIAL_CASH = 1000
BASE_CURRENCY = 'EUR'
DEFAULT_ASSET = 'EUTEUR'
DEFAULT_DATA_PATH = 'D:\\StableTrade_dataset\\EUTEUR_1m\\EUTEUR_1m_final_merged.csv'
DEFAULT_METRICS_PATH = 'metrics_data.json'

def verify_files_exist(assets: Dict[str, str], logger: logging.Logger) -> bool:
    all_files_exist = True
    for asset_name, file_path in assets.items():
        if not os.path.exists(file_path):
            logger.error(f"CSV file for {asset_name} does not exist at path: {file_path}")
            all_files_exist = False

    if all_files_exist:
        logger.info("All asset files verified successfully.")
    
    return all_files_exist

def get_user_input(prompt, default=None):
    user_input = input(f"{prompt} [{default}]: ").strip() if default else input(f"{prompt}: ").strip()
    return user_input if user_input else default

def run_backtest(custom: bool, log_levels: Optional[Dict[str, str]] = None):
    if log_levels is None:
        log_levels = {}

    set_log_levels(log_levels)
    main_logger = setup_logger('main')
    main_logger.warning("Starting the backtesting process.")

    if custom:
        asset = get_user_input("Enter asset name", DEFAULT_ASSET)
        data_path = get_user_input("Enter data file path", DEFAULT_DATA_PATH)
        initial_cash = float(get_user_input("Enter initial cash", str(INITIAL_CASH)))
        depeg_threshold = float(get_user_input("Enter depeg threshold", "8"))
        trade_amount = float(get_user_input("Enter trade amount", "0.1"))
        stop_loss = float(get_user_input("Enter stop loss", "0.02"))
        take_profit = float(get_user_input("Enter take profit", "0.05"))
        trailing_stop = float(get_user_input("Enter trailing stop", "0.01"))
        slippage = float(get_user_input("Enter slippage", "0.0001"))
    else:
        asset = DEFAULT_ASSET
        data_path = DEFAULT_DATA_PATH
        initial_cash = INITIAL_CASH
        depeg_threshold = 8
        trade_amount = 0.2
        stop_loss = 0.9
        take_profit = 0.2
        trailing_stop = None
        slippage = 0.0001

    assets = {asset: data_path}

    if not verify_files_exist(assets, main_logger):
        main_logger.error("File verification failed. Exiting the process.")
        return

    trade_manager = TradeManager(base_currency=BASE_CURRENCY)
    signal_database = SignalDatabase(trade_manager)
    main_logger.warning("Trade manager and signal database initialized.")

    strategies = {
        asset: DepegStrategy(
            market=asset,
            trade_manager=trade_manager,
            depeg_threshold=depeg_threshold,
            trade_amount=trade_amount,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop
        )
    }
    main_logger.warning(f"Strategies initialized for {', '.join(strategies.keys())}.")

    portfolio_config = {
        asset_name: {
            'market_type': strategy.config['market_type'],
            'fees': strategy.config['fees'],
            'max_trades': strategy.config['max_trades'],
            'max_exposure': strategy.config['max_exposure']
        } for asset_name, strategy in strategies.items()
    }

    portfolio = Portfolio(
        initial_cash=initial_cash,
        portfolio_config=portfolio_config,
        signal_database=signal_database,
        trade_manager=trade_manager,
        base_currency=BASE_CURRENCY
    )
    main_logger.warning(f"Portfolio initialized with {initial_cash} {BASE_CURRENCY}.")

    metrics_module = MetricsModule(market_data_path=DEFAULT_DATA_PATH, base_currency=BASE_CURRENCY)
    main_logger.warning("Metrics module initialized.")

    backtest_engine = BacktestEngine(
        assets=assets,
        strategies=strategies,
        portfolio=portfolio,
        trade_manager=trade_manager,
        base_currency=BASE_CURRENCY,
        slippage=slippage,
        metrics=metrics_module,
        signal_database=signal_database
    )
    main_logger.warning("Backtest engine initialized.")

    try:
        main_logger.warning("Starting data preprocessing...")
        backtest_engine.preprocess_data()
        main_logger.warning("Data preprocessing completed successfully.")

        main_logger.warning("Starting backtest execution...")
        backtest_engine.run_backtest()
        main_logger.warning("Backtest completed successfully.")

    except Exception as e:
        main_logger.error(f"An error occurred during the backtest: {e}", exc_info=True)
    
    main_logger.warning("Backtesting process completed.")

def analyze_results(custom: bool):
    if custom:
        metrics_file = get_user_input("Enter path of market data file", DEFAULT_DATA_PATH)
    else:
        metrics_file = DEFAULT_DATA_PATH
    try:
        metrics_module = MetricsModule(market_data_path=metrics_file, base_currency=BASE_CURRENCY)
        metrics_module.run()
    except Exception as e:
        print(f"Error analyzing results: {str(e)}")

def main():
    while True:
        print("\nSelect an option:")
        print("1. Run Backtest (Default Settings)")
        print("2. Run Backtest (Custom Settings)")
        print("3. Analyze Results (Default File)")
        print("4. Analyze Results (Custom File)")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        custom_log_levels = {
            'main': 'WARNING',
            'trade_manager': 'WARNING',
            'signal_database': 'WARNING',
            'depeg_strategy': 'WARNING',
            'portfolio': 'WARNING',
            'metrics': 'INFO',
            'backtest_engine': 'WARNING'
        }

        if choice == '1':
            run_backtest(custom=False, log_levels=custom_log_levels)
        elif choice == '2':
            run_backtest(custom=True, log_levels=custom_log_levels)
        elif choice == '3':
            analyze_results(custom=False)
        elif choice == '4':
            analyze_results(custom=True)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()