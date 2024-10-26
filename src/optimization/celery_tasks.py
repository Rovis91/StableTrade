"""Task definitions for optimization processing."""

from celery import Task, Celery
from typing import Dict, Any, Optional, Type
import time
import logging
from pathlib import Path
import importlib
from src.logger import setup_logger

# Import necessary base components
from src.portfolio import Portfolio
from src.trade_manager import TradeManager
from src.signal_database import SignalDatabase
from src.metrics import MetricsModule
from src.backtest_engine import BacktestEngine
from src.strategy.base_strategy import Strategy

# Initialize Celery app
celery_app = Celery(
    'optimization_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

class OptimizationTask(Task):
    """Base task class with error handling and logging."""
    
    abstract = True
    
    def __init__(self):
        self.logger = setup_logger('optimization_task')

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        self.logger.error(
            f"Task {task_id} failed: {exc}\nArgs: {args}\nKwargs: {kwargs}\n"
            f"Traceback: {einfo}"
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        self.logger.warning(
            f"Task {task_id} retrying: {exc}\nArgs: {args}\nKwargs: {kwargs}"
        )


@celery_app.task(base=OptimizationTask, bind=True, max_retries=3)
def run_optimization(
    self,
    strategy_module: str,
    strategy_name: str,
    params: Dict[str, Any],
    data_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run optimization task with the given parameters and strategy.
    
    Args:
        strategy_module: Module path of the strategy
        strategy_name: Name of the strategy class
        params: Parameter combination to test
        data_path: Path to market data
        config: Additional configuration
    """
    try:
        self.logger = setup_logger('optimization_task')
        task_id = self.request.id
        self.logger.info(f"Starting task {task_id} with params: {params}")

        
        # Dynamically import strategy class
        module = importlib.import_module(strategy_module)
        strategy_class = getattr(module, strategy_name)
        
        # Initialize components
        from src.trade_manager import TradeManager
        from src.signal_database import SignalDatabase
        from src.metrics import MetricsModule
        
        trade_manager = TradeManager(base_currency=config['base_currency'])
        signal_database = SignalDatabase(trade_manager)
        
        # Create strategy instance
        strategy = strategy_class(
            market=config['asset'],
            trade_manager=trade_manager,
            **params
        )
        
        # Initialize metrics module
        metrics_module = MetricsModule(
            market_data_path=data_path,
            base_currency=config['base_currency']
        )
        portfolio_config = {
            config['asset']: {
                'market_type': strategy.config['market_type'],
                'fees': strategy.config['fees'],
                'max_trades': strategy.config['max_trades'],
                'max_exposure': strategy.config['max_exposure']
            }
        }
        
        portfolio = Portfolio(
            initial_cash=config['initial_cash'],
            portfolio_config=portfolio_config,
            signal_database=signal_database,
            trade_manager=trade_manager,
            base_currency=config['base_currency']
        )
        # Run backtest
        from src.backtest_engine import BacktestEngine
        backtest_engine = BacktestEngine(
            assets={config['asset']: data_path},
            strategies={config['asset']: strategy},
            portfolio=portfolio,
            trade_manager=trade_manager,
            base_currency=config['base_currency'],
            slippage=config.get('slippage', 0),
            metrics=metrics_module,
            signal_database=signal_database
        )
        
        # Execute backtest
        start_time = time.time()
        backtest_engine.preprocess_data()
        backtest_engine.run_backtest()
        
        # Get metrics summary
        metrics_summary = metrics_module.get_summary()
        
        self.logger.info(f"Task {task_id} preprocessing data...")
        backtest_engine.preprocess_data()
        
        self.logger.info(f"Task {task_id} running backtest...")
        backtest_engine.run_backtest()
        
        self.logger.info(f"Task {task_id} getting metrics...")
        metrics_summary = metrics_module.get_summary()
        
        result = {
            'parameters': params,
            'metrics': metrics_summary,
            'execution_time': time.time() - start_time
        }

        
        self.logger.info(f"Optimization completed in {result['execution_time']:.2f} seconds")
        return result
        
    except Exception as e:
        self.logger.error(f"Error during optimization: {str(e)}")
        raise self.retry(exc=e, countdown=2**self.request.retries)
    
# Celery configuration
celery_app.conf.update({
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
    'result_expires': 3600,  # Results expire after 1 hour
    'task_time_limit': 3600,  # Tasks timeout after 1 hour
    'task_soft_time_limit': 3300,  # Soft timeout after 55 minutes
    'worker_max_tasks_per_child': 50,
    'worker_max_memory_per_child': 200000  # Restart worker after 200MB
})