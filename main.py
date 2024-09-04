import argparse
import pandas as pd
from src.data_loader import load_csv
from src.optimization import generate_parameter_grid, run_grid_search
from src.utils import save_results_to_json

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run StableTrade backtest.')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--balance', type=float, default=1000, help='Initial balance')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission per trade')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    args = parser.parse_args()

    # Load data
    df = load_csv(args.data)

    # Define parameter ranges for grid search
    param_ranges = {
        'offset': [0.01, 0.02, 0.03],
        'neutral_period': [1440, 2880],
        'day_period': [360, 720]
    }

    # Generate parameter grid
    param_grid = generate_parameter_grid(param_ranges)

    # Run grid search (distributed via Celery)
    result_group = run_grid_search(df, args.balance, args.commission, param_grid)

    # Wait for all results to complete
    results = result_group.get()

    # Collect and structure the final results
    all_results = []
    for result in results:
        all_results.append({
            "final_balance": result['final_balance'],
            "cumulative_return": result['cumulative_return'],
            "max_drawdown": result['max_drawdown'],
            "sharpe_ratio": result['sharpe_ratio'],
            "profit_factor": result['profit_factor'],
            "total_trades": result['total_trades'],
            "trade_log": result['trade_log']
        })

    # Save the aggregated results to JSON
    save_results_to_json(args.output, all_results)

if __name__ == "__main__":
    main()
