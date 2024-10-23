"""
celery_tasks.py

This module handles distributed backtesting using Celery.
It defines tasks for running individual parameter combinations
and manages the parallel execution of optimization runs.
"""

from celery import Celery, group, chord
from typing import Dict, List, Any
import logging
from pathlib import Path
import json
import time

from src.backtest_engine import BacktestEngine
from src.portfolio import Portfolio
from src.trade_manager import TradeManager
from src.strategy.depeg_strategy import DepegStrategy
from src.signal_database import SignalDatabase
from src.metrics import MetricsModule

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery('optimization_tasks',
                    broker='redis://localhost:6379/0',
                    backend='redis://localhost:6379/0')

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour time limit per task
    task_soft_time_limit=3300,  # Soft limit 55 minutes
    worker_prefetch_multiplier=1,  # Each worker takes one task at a time
    task_acks_late=True  # Tasks are acknowledged after completion
)

@celery_app.task(bind=True, name='run_backtest')
def run_backtest(self, param_combination: Dict[str, float], base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single backtest with given parameters.

    Args:
        param_combination: Dictionary of parameter values to test
        base_config: Base configuration for the backtest

    Returns:
        Dictionary containing parameters and their corresponding metrics or error
    """
    try:
        logger.info(f"Starting backtest with parameters: {param_combination}")
        start_time = time.time()

        # Initialize components for this backtest
        trade_manager = TradeManager(base_currency=base_config['base_currency'])
        signal_database = SignalDatabase(trade_manager)

        strategy = DepegStrategy(
            market=base_config['asset'],
            trade_manager=trade_manager,
            depeg_threshold=param_combination['depeg_threshold'],
            trade_amount=param_combination['trade_amount'],
            stop_loss=param_combination.get('stop_loss'),
            take_profit=param_combination.get('take_profit'),
            trailing_stop=param_combination.get('trailing_stop')
        )

        portfolio_config = {
            base_config['asset']: {
                'market_type': strategy.config['market_type'],
                'fees': strategy.config['fees'],
                'max_trades': strategy.config['max_trades'],
                'max_exposure': strategy.config['max_exposure']
            }
        }

        portfolio = Portfolio(
            initial_cash=base_config['initial_cash'],
            portfolio_config=portfolio_config,
            signal_database=signal_database,
            trade_manager=trade_manager,
            base_currency=base_config['base_currency']
        )

        metrics_module = MetricsModule(
            market_data_path=base_config['data_path'],
            base_currency=base_config['base_currency']
        )

        # Run backtest
        backtest_engine = BacktestEngine(
            assets={base_config['asset']: base_config['data_path']},
            strategies={base_config['asset']: strategy},
            portfolio=portfolio,
            trade_manager=trade_manager,
            base_currency=base_config['base_currency'],
            slippage=base_config.get('slippage', 0),
            metrics=metrics_module,
            signal_database=signal_database
        )

        backtest_engine.preprocess_data()
        backtest_engine.run_backtest()

        metrics_summary = metrics_module.get_summary()

        result = {
            'parameters': param_combination,
            'metrics': metrics_summary,
            'execution_time': time.time() - start_time
        }

        logger.info(f"Completed backtest with parameters: {param_combination}")
        return result

    except Exception as e:
        logger.error(f"Error in backtest task: {str(e)}", exc_info=True)
        return {
            'parameters': param_combination,
            'error': str(e),
            'status': 'failed'
        }

@celery_app.task(name='process_optimization_results')
def process_optimization_results(results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    """
    Process results from all backtests, summarize, and save them.

    Args:
        results: List of results from individual backtests
        output_dir: Directory to save results

    Returns:
        Summary of optimization results
    """
    try:
        logger.info("Processing optimization results")

        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]

        summary = {
            'total_runs': len(results),
            'successful_runs': len(successful_results),
            'failed_runs': len(failed_results),
            'best_results': {},
            'execution_statistics': {
                'total_time': sum(r.get('execution_time', 0) for r in successful_results),
                'average_time': sum(r.get('execution_time', 0) for r in successful_results) / len(successful_results) if successful_results else 0
            }
        }

        # Find best results for each metric
        if successful_results:
            metrics = successful_results[0]['metrics'].keys()
            for metric in metrics:
                best_result = max(successful_results, key=lambda x: x['metrics'].get(metric, float('-inf')))
                summary['best_results'][metric] = {
                    'value': best_result['metrics'][metric],
                    'parameters': best_result['parameters']
                }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / 'optimization_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        with open(output_path / 'optimization_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)

        if failed_results:
            with open(output_path / 'failed_runs.json', 'w') as f:
                json.dump(failed_results, f, indent=4)
            logger.warning(f"{len(failed_results)} runs failed. See failed_runs.json for details")

        logger.info("Completed processing optimization results")
        return summary

    except Exception as e:
        logger.error(f"Error processing optimization results: {str(e)}", exc_info=True)
        raise

def create_optimization_tasks(param_combinations: List[Dict[str, float]], base_config: Dict[str, Any], output_dir: str) -> group:
    """
    Create a group of Celery tasks for parameter optimization.

    Args:
        param_combinations: List of parameter combinations to test
        base_config: Base configuration for backtests
        output_dir: Directory to save results

    Returns:
        Celery group of tasks
    """
    try:
        logger.info(f"Creating optimization tasks for {len(param_combinations)} combinations")

        backtest_tasks = [run_backtest.s(params, base_config) for params in param_combinations]

        workflow = chord(backtest_tasks, process_optimization_results.s(output_dir))

        logger.info("Successfully created optimization tasks")
        return workflow

    except Exception as e:
        logger.error(f"Error creating optimization tasks: {str(e)}", exc_info=True)
        raise

def monitor_task_progress(task_group):
    """
    Monitor the progress of Celery tasks in a task group.

    Args:
        task_group: Celery group of tasks to monitor
    """
    try:
        total_tasks = len(task_group.tasks)
        completed, failed = 0, 0

        while completed + failed < total_tasks:
            time.sleep(5)  # Check progress every 5 seconds
            completed = sum(1 for task in task_group.tasks if task.status == 'SUCCESS')
            failed = sum(1 for task in task_group.tasks if task.status == 'FAILURE')

            logger.info(f"Progress: {completed}/{total_tasks} completed, {failed} failed")

        logger.info(f"All tasks completed. Successful: {completed}, Failed: {failed}")

    except Exception as e:
        logger.error(f"Error monitoring tasks: {str(e)}", exc_info=True)
        raise
