from src.logger import setup_logger, set_log_levels
from src.optimization.optimizer import StrategyOptimizer
from pathlib import Path

# Set less restrictive log levels for debugging
custom_log_levels = {
    'main': 'INFO',
    'trade_manager': 'WARNING',
    'signal_database': 'WARNING',
    'depeg_strategy': 'WARNING',
    'portfolio': 'WARNING',
    'metrics': 'WARNING',
    'backtest_engine': 'WARNING',
    'data_preprocessor': 'WARNING',
    'optimization': 'INFO'
}

set_log_levels(custom_log_levels)
logger = setup_logger('main')

def main():
    """
    Main function to initialize and run the optimization process.
    
    This function sets up the parameter ranges, initializes the optimizer,
    estimates the runtime, and executes the optimization of trading strategies.
    """
    try:
        param_ranges = {
            'depeg_threshold': {'start': 1, 'end': 3, 'step': 1},
            'trade_amount': {'start': 0.1, 'end': 0.1, 'step': 0.1},
            'stop_loss': {'start': 0.02, 'end': 0.02, 'step': 0.02},
            'take_profit': {'start': 0.05, 'end': 0.05, 'step': 0.05},
            'trailing_stop': {'start': 0.005, 'end': 0.01, 'step': 0.005}
        }

        base_config = {
            'asset': 'EUTEUR',
            'data_path': 'D:\\StableTrade_dataset\\EUTEUR_1m\\EUTEUR_1m_final_merged.csv',
            'initial_cash': 1000,
            'base_currency': 'EUR',
            'slippage': 0.0001
        }

        # Initialize optimizer
        print("\nInitializing optimizer...")
        logger.info("Initializing optimizer with provided parameter ranges and base configuration.")
        optimizer = StrategyOptimizer(
            param_ranges=param_ranges,
            base_config=base_config,
            output_dir='optimization_results'
        )

        # Get runtime estimate
        estimate = optimizer.estimate_runtime(single_run_time=60)
        logger.info(f"Estimated runtime for {estimate['total_combinations']} combinations: "
                    f"{estimate['estimated_time']['hours']:.2f} hours")
        print(f"\nStarting optimization of {estimate['total_combinations']} combinations")
        print(f"Estimated runtime: {estimate['estimated_time']['hours']:.2f} hours\n")

        # Run optimization
        print("Starting backtests...")
        logger.info("Starting the optimization process.")
        summary = optimizer.run_optimization(test_run=False)  # Set to True for testing with 3 combinations
        
        # Print final results
        print("\nOptimization completed!")
        logger.info("Optimization process completed successfully.")
        results_dir = Path(summary['file_locations']['base_directory']).absolute()
        print(f"Results saved to: {results_dir}")
        print(f"Total runtime: {summary['runtime']['total_seconds']/3600:.2f} hours")

        # Print some basic statistics
        if 'portfolio_metrics' in summary:
            print("\nBest Results:")
            print(f"Total Return: {summary['portfolio_metrics'].get('total_return', 'N/A')}%")
            print(f"Sharpe Ratio: {summary['portfolio_metrics'].get('sharpe_ratio', 'N/A')}")
            print(f"Max Drawdown: {summary['portfolio_metrics'].get('max_drawdown', 'N/A')}%")

    except Exception as e:
        logger.error(f"An error occurred during the optimization process: {str(e)}")
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
