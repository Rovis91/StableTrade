import time
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .parameter_grid import ParameterGrid
from .result_manager import OptimizationResults
from .visualizer import OptimizationVisualizer
from ..backtest_engine import BacktestEngine
from ..portfolio import Portfolio
from ..trade_manager import TradeManager
from ..strategy.depeg_strategy import DepegStrategy
from ..signal_database import SignalDatabase
from ..metrics import MetricsModule

# Configure logger
logger = logging.getLogger(__name__)


class StrategyOptimizer:
    def __init__(self, 
                 param_ranges: Dict[str, Dict[str, float]],
                 base_config: Dict[str, Any],
                 output_dir: Optional[str] = None):
        """
        Initialize the strategy optimizer.

        Args:
            param_ranges (Dict[str, Dict[str, float]]): Ranges for strategy parameters.
            base_config (Dict[str, Any]): Base configuration for the backtest.
            output_dir (Optional[str]): Directory to save optimization results.
        """
        self.validate_config(base_config)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir or f"optimization_results_{timestamp}")
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = self.output_dir / 'optimization_report'
        self.plots_dir = self.output_dir / 'plots'
        self.report_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        self.param_grid = ParameterGrid(param_ranges)
        self.results = OptimizationResults()
        self.visualizer = OptimizationVisualizer(self.results, str(self.plots_dir))
        self.base_config = base_config
        
        logger.info(f"Initialized StrategyOptimizer with {self.param_grid.total_combinations} combinations")
        logger.info(f"Output directory structure created at {self.output_dir}")

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate the base configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        required_fields = ['asset', 'data_path', 'initial_cash', 'base_currency']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
                
        if not Path(config['data_path']).exists():
            raise ValueError(f"Data file not found: {config['data_path']}")
            
        if config['initial_cash'] <= 0:
            raise ValueError("Initial cash must be positive")

    def run_single_backtest(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Run a single backtest with given parameters.

        Args:
            params (Dict[str, float]): Parameters for the backtest.

        Returns:
            Dict[str, Any]: Backtest result including metrics or error.
        """
        try:
            # Initialize components
            trade_manager = TradeManager(base_currency=self.base_config['base_currency'])
            signal_database = SignalDatabase(trade_manager)

            # Create strategy with parameters
            strategy = DepegStrategy(
                market=self.base_config['asset'],
                trade_manager=trade_manager,
                depeg_threshold=params['depeg_threshold'],
                trade_amount=params['trade_amount'],
                stop_loss=params.get('stop_loss'),
                take_profit=params.get('take_profit'),
                trailing_stop=params.get('trailing_stop')
            )

            portfolio_config = {
                self.base_config['asset']: {
                    'market_type': strategy.config['market_type'],
                    'fees': strategy.config['fees'],
                    'max_trades': strategy.config['max_trades'],
                    'max_exposure': strategy.config['max_exposure']
                }
            }

            portfolio = Portfolio(
                initial_cash=self.base_config['initial_cash'],
                portfolio_config=portfolio_config,
                signal_database=signal_database,
                trade_manager=trade_manager,
                base_currency=self.base_config['base_currency']
            )

            metrics_module = MetricsModule(
                market_data_path=self.base_config['data_path'],
                base_currency=self.base_config['base_currency'],
                metric_groups='basic',  
                silent=True 
            )

            backtest_engine = BacktestEngine(
                assets={self.base_config['asset']: self.base_config['data_path']},
                strategies={self.base_config['asset']: strategy},
                portfolio=portfolio,
                trade_manager=trade_manager,
                base_currency=self.base_config['base_currency'],
                slippage=self.base_config.get('slippage', 0),
                metrics=metrics_module,
                signal_database=signal_database
            )

            # Run backtest
            backtest_engine.preprocess_data()
            backtest_engine.run_backtest()

            # Get metrics
            metrics = metrics_module.run()

            result = {
                'parameters': params,
                'metrics': metrics,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            return result
        
        except Exception as e:
            print(f"\nError in backtest with params {params}: {str(e)}")  # Debug output
            return {
                'parameters': params,
                'error': str(e)
            }

    def run_optimization(self, test_run: bool = False) -> Dict[str, Any]:
        """
        Run the full optimization process.

        Args:
            test_run (bool): Whether to run a test with limited combinations.

        Returns:
            Dict[str, Any]: Summary of the optimization.
        """
        try:
            start_time = time.time()
            print("\nStarting optimization process...")  # Debug output

            # Get parameter combinations
            combinations = self.param_grid.generate_combinations()
            if test_run:
                combinations = combinations[:3]
                print(f"Test run with {len(combinations)} combinations")  # Debug output
            else:
                print(f"Full run with {len(combinations)} combinations")  # Debug output

            # Setup progress bar
            total_combinations = len(combinations)
            progress_bar = tqdm(
                total=total_combinations,
                desc="Optimizing",
                unit="test",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )

            results = []
            successful = 0
            failed = 0

            # Run backtests
            for i, params in enumerate(combinations):
                print(f"\nRunning backtest {i+1}/{total_combinations} with params: {params}")  # Debug output
                result = self.run_single_backtest(params)
                
                if 'error' not in result:
                    successful += 1
                    print(f"Backtest successful")  # Debug output
                    self.results.add_result(
                        params=result['parameters'],
                        metrics=result['metrics']
                    )
                else:
                    failed += 1
                    print(f"Backtest failed: {result.get('error', 'Unknown error')}")  # Debug output
                
                results.append(result)
                
                # Update progress
                progress_bar.set_postfix({
                    'success': f"{successful}/{total_combinations}",
                    'failed': failed
                }, refresh=True)
                progress_bar.update(1)

            progress_bar.close()
            print("\nAll backtests completed")  # Debug output

            # Generate reports
            print("Generating reports...")  # Debug output
            self._generate_visualization_report()
            summary = self._create_optimization_summary(start_time, results)
            self.save_results()

            return summary

        except Exception as e:
            print(f"\nOptimization error: {str(e)}")  # Debug output
            if 'progress_bar' in locals():
                progress_bar.close()
            raise

    def _generate_visualization_report(self) -> None:
        """Generate visualization plots."""
        try:
            if self.results.results.empty:
                logger.warning("No results available for visualization")
                return
                
            logger.info("Generating visualization report")
            self.visualizer.create_optimization_report()
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")

    def _create_optimization_summary(self, start_time: float, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create optimization summary.

        Args:
            start_time (float): The time the optimization started.
            results (List[Dict[str, Any]]): The results of the optimization.

        Returns:
            Dict[str, Any]: Summary of the optimization results.
        """
        try:
            summary = {
                'total_runs': len(results),
                'successful_runs': len([r for r in results if 'error' not in r]),
                'failed_runs': len([r for r in results if 'error' in r]),
                'best_results': {},
                'parameters': self.param_grid.get_param_info(),
                'runtime': {
                    'total_seconds': time.time() - start_time,
                    'average_per_run': (time.time() - start_time) / len(results) if results else 0
                },
                'file_locations': {
                    'base_directory': str(self.output_dir),
                    'report_directory': str(self.report_dir),
                    'plots_directory': str(self.plots_dir)
                }
            }

            # Find best results for each metric
            successful_results = [r for r in results if 'error' not in r]
            if successful_results:
                metrics = successful_results[0]['metrics'].keys()
                for metric in metrics:
                    best_result = max(successful_results, 
                                    key=lambda x: x['metrics'].get(metric, float('-inf')))
                    summary['best_results'][metric] = {
                        'value': best_result['metrics'][metric],
                        'parameters': best_result['parameters']
                    }

            # Save summary to the report directory
            summary_path = self.report_dir / 'optimization_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)

            # Save failed runs separately if any exist
            failed_results = [r for r in results if 'error' in r]
            if failed_results:
                failed_path = self.report_dir / 'failed_runs.json'
                with open(failed_path, 'w') as f:
                    json.dump(failed_results, f, indent=4)

            return summary

        except Exception as e:
            logger.error(f"Error creating optimization summary: {str(e)}")
            return {}

    def estimate_runtime(self, single_run_time: float = 60) -> Dict[str, Any]:
        """
        Estimate the total runtime for optimization.

        Args:
            single_run_time (float): Estimated time for a single backtest in seconds.

        Returns:
            Dict[str, Any]: Runtime estimates including hours, minutes, and seconds.
        """
        total_seconds = self.param_grid.total_combinations * single_run_time
        
        hours = total_seconds / 3600
        minutes = (total_seconds % 3600) / 60
        seconds = total_seconds % 60

        return {
            'total_combinations': self.param_grid.total_combinations,
            'total_seconds': total_seconds,
            'estimated_time': {
                'hours': hours,
                'minutes': minutes,
                'seconds': seconds
            }
        }

    def save_results(self) -> None:
        """Save results to CSV in the report directory."""
        try:
            results_path = self.report_dir / 'optimization_results.csv'
            self.results.save_to_csv(str(results_path))
            logger.info(f"Saved results to {results_path}")

            # Save parameter combinations
            param_path = self.report_dir / 'parameter_combinations.json'
            with open(param_path, 'w') as f:
                json.dump(self.param_grid.get_param_info(), f, indent=4)
            logger.info(f"Saved parameter combinations to {param_path}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            raise