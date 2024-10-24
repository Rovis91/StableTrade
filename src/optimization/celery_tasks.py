import sys
from pathlib import Path

# Add the project root to the system path for imports
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules and packages
from celery import Celery, group
from typing import Dict, List, Any
import logging
import json
import time

from ..backtest_engine import BacktestEngine
from ..portfolio import Portfolio
from ..trade_manager import TradeManager
from ..strategy.depeg_strategy import DepegStrategy
from ..signal_database import SignalDatabase
from ..metrics import MetricsModule

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Celery with Redis as the broker and backend
celery_app = Celery(
    'optimization_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1-hour time limit per task
    task_soft_time_limit=3300,  # Soft limit of 55 minutes
    worker_prefetch_multiplier=1,  # Each worker takes one task at a time
    task_acks_late=True,  # Tasks are acknowledged after completion
    result_backend='redis://localhost:6379/0',  # Ensure this matches your Redis setup
    task_ignore_result=False,  # We need to track results
    task_always_eager=False,  # Run tasks asynchronously
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
    worker_max_memory_per_child=200000  # Restart worker if memory usage exceeds 200MB
)

@celery_app.task(bind=True, name='run_backtest')
def run_backtest(self, param_combination: Dict[str, float], base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single backtest with given parameters.

    Args:
        param_combination (Dict[str, float]): Dictionary of parameter values to test
        base_config (Dict[str, Any]): Base configuration for the backtest

    Returns:
        Dict[str, Any]: Parameters and their corresponding metrics or error
    """
    try:
        task_id = self.request.id
        logger.info(f"Starting backtest {task_id} with parameters: {param_combination}")
        start_time = time.time()

        # Initialize components for this backtest
        trade_manager = TradeManager(base_currency=base_config['base_currency'])
        signal_database = SignalDatabase(trade_manager)

        # Initialize the strategy
        strategy = DepegStrategy(
            market=base_config['asset'],
            trade_manager=trade_manager,
            depeg_threshold=param_combination['depeg_threshold'],
            trade_amount=param_combination['trade_amount'],
            stop_loss=param_combination.get('stop_loss'),
            take_profit=param_combination.get('take_profit'),
            trailing_stop=param_combination.get('trailing_stop')
        )

        # Setup portfolio configuration
        portfolio_config = {
            base_config['asset']: {
                'market_type': strategy.config['market_type'],
                'fees': strategy.config['fees'],
                'max_trades': strategy.config['max_trades'],
                'max_exposure': strategy.config['max_exposure']
            }
        }

        # Initialize the portfolio
        portfolio = Portfolio(
            initial_cash=base_config['initial_cash'],
            portfolio_config=portfolio_config,
            signal_database=signal_database,
            trade_manager=trade_manager,
            base_currency=base_config['base_currency']
        )

        # Initialize the metrics module
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

        # Get metrics summary
        metrics_summary = metrics_module.get_summary()

        result = {
            'parameters': param_combination,
            'metrics': metrics_summary,
            'execution_time': time.time() - start_time
        }
        duration = time.time() - start_time
        logger.info(f"Completed backtest {task_id} in {duration:.2f} seconds")
        return result

    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {str(e)}", exc_info=True)
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
        results (List[Dict[str, Any]]): List of results from individual backtests
        output_dir (str): Directory to save results

    Returns:
        Dict[str, Any]: Summary of optimization results
    """
    try:
        logger.info(f"Processing {len(results)} optimization results")

        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]

        logger.info(f"Successfully processed {len(successful_results)} results")
        if failed_results:
            logger.warning(f"Found {len(failed_results)} failed results")

        summary = {
            'total_runs': len(results),
            'successful_runs': len(successful_results),
            'failed_runs': len(failed_results),
            'best_results': {},
            'execution_statistics': {
                'total_time': sum(r.get('execution_time', 0) for r in successful_results),
                'average_time': (
                    sum(r.get('execution_time', 0) for r in successful_results) /
                    len(successful_results) if successful_results else 0
                )
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
        logger.error(f"Error processing results: {str(e)}", exc_info=True)
        raise


def create_optimization_tasks(param_combinations: List[Dict[str, float]], base_config: Dict[str, Any], output_dir: str):
    """
    Create a group of Celery tasks for parameter optimization.

    Args:
        param_combinations (List[Dict[str, float]]): List of parameter combinations to test
        base_config (Dict[str, Any]): Base configuration for backtests
        output_dir (str): Directory to save results

    Returns:
        AsyncResult: Async result from the group of tasks
    """
    try:
        total_combinations = len(param_combinations)
        logger.info(f"Creating optimization tasks for {total_combinations} combinations")

        # Create tasks in smaller chunks to avoid memory issues
        chunk_size = 1000
        all_tasks = []

        for i in range(0, total_combinations, chunk_size):
            chunk = param_combinations[i:i + chunk_size]
            tasks = [run_backtest.s(params, base_config) for params in chunk]
            all_tasks.extend(tasks)
            logger.info(f"Created tasks {i} to {min(i + chunk_size, total_combinations)}")

        # Create workflow with chunks
        workflow = group(all_tasks) | process_optimization_results.s(output_dir)

        # Start the workflow
        result = workflow.apply_async()
        logger.info("Started optimization workflow")

        return result

    except Exception as e:
        logger.error(f"Error creating optimization tasks: {str(e)}", exc_info=True)
        raise


def monitor_task_progress(task_group_result):
    """
    Monitor the progress of Celery tasks.

    Args:
        task_group_result (AsyncResult): Async result from a group of tasks
    """
    try:
        logger.info("Starting task monitoring")

        if not task_group_result:
            logger.error("No task result to monitor")
            return

        while not task_group_result.ready():
            # Get all subtasks
            if hasattr(task_group_result, 'children'):
                subtasks = task_group_result.children
            else:
                subtasks = [task_group_result]

            total = len(subtasks)
            completed = sum(1 for task in subtasks if task.ready())
            failed = sum(1 for task in subtasks if task.failed())
            pending = total - completed

            logger.info(f"Progress: {completed}/{total} completed, {failed} failed, {pending} pending")

            # Calculate and log estimated time remaining
            if completed > 0:
                time_elapsed = time.time() - task_group_result.timestamp
                time_per_task = time_elapsed / completed
                time_remaining = time_per_task * pending
                hours_remaining = time_remaining / 3600
                logger.info(f"Estimated time remaining: {hours_remaining:.2f} hours")

            time.sleep(10)  # Check every 10 seconds

        logger.info("All tasks completed!")

    except Exception as e:
        logger.error(f"Error monitoring tasks: {str(e)}", exc_info=True)
        raise
