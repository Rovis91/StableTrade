import json
import numpy as np
from backtest import backtest
from src.metrics import (
    calculate_sharpe_ratio,
    calculate_cumulative_return,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_total_trades
)

def run_backtest_and_save_results(df, initial_balance, commission, file_path):
    """
    Run the backtest, calculate performance metrics, and save results to a JSON file.
    
    :param df: DataFrame containing the historical data and signals.
    :param initial_balance: Starting capital for the backtest.
    :param commission: Commission per trade.
    :param file_path: Path to save the results JSON file.
    """
    # Run the backtest
    final_balance, trades, trade_log = backtest(df, initial_balance, commission)
    
    # Create a balance series for calculating max drawdown and Sharpe ratio
    balance_series = [initial_balance]  # We would build this in the backtest loop as we update the balance

    # Calculate metrics
    cumulative_return = calculate_cumulative_return(initial_balance, final_balance)
    max_drawdown = calculate_max_drawdown(balance_series)
    sharpe_ratio = calculate_sharpe_ratio(np.array(balance_series) / initial_balance - 1)
    profit_factor = calculate_profit_factor(trade_log)
    total_trades = calculate_total_trades(trade_log)

    # Prepare the results dictionary with all metrics
    results = {
        "final_balance": final_balance,
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "trade_log": trade_log
    }

    # Save the results to JSON
    save_results_to_json(file_path, results)

def save_results_to_json(file_path, results):
    """
    Save the backtest results and metrics to a JSON file.
    
    :param file_path: Path to the output JSON file.
    :param results: Dictionary containing backtest results and metrics.
    """
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
