from itertools import product
from celery import group
from src.optimization.result_manager import OptimizationResults

def generate_parameter_grid(param_ranges):
    """
    Generates a grid of parameters for optimization.
    
    :param param_ranges: Dictionary where keys are parameter names and values are lists of possible values
    :return: List of dictionaries with parameter combinations
    """
    keys, values = zip(*param_ranges.items())
    return [dict(zip(keys, v)) for v in product(*values)]

def run_grid_search(df, initial_balance, commission, param_grid):
    """
    Run a grid search over the parameter combinations using Celery.
    
    :param df: Historical data
    :param initial_balance: Starting capital
    :param commission: Commission per trade
    :param param_grid: List of parameter combinations to test
    :return: Async result group for tracking task progress
    """
    task_group = group(run_backtest_task.s(df.to_dict(), initial_balance, commission, params) for params in param_grid)
    return task_group.apply_async()
