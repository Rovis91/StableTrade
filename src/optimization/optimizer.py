import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json
from .parameter_grid import ParameterGrid
from .result_manager import OptimizationResults
from .visualizer import OptimizationVisualizer
from .celery_tasks import create_optimization_tasks, monitor_task_progress

logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    Coordinates optimization components including parameter grid generation,
    parallel backtesting, results collection, and visualization generation.

    Attributes:
        param_grid: ParameterGrid instance for generating combinations
        results: OptimizationResults instance for managing results
        visualizer: OptimizationVisualizer instance for creating plots
        base_config: Base configuration for backtests
        output_dir: Directory for saving results and plots
    """
    
    def __init__(self, 
                 param_ranges: Dict[str, Dict[str, float]],
                 base_config: Dict[str, Any],
                 output_dir: Optional[str] = None):
        """
        Initialize the strategy optimizer.

        Args:
            param_ranges: Dictionary of parameter ranges for optimization
            base_config: Base configuration for backtests
            output_dir: Directory for saving results (optional)
        """
        self.validate_config(base_config)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir or f"optimization_results_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.param_grid = ParameterGrid(param_ranges)
        self.results = OptimizationResults()
        self.visualizer = OptimizationVisualizer(self.results, str(self.output_dir))
        self.base_config = base_config
        
        logger.info(f"Initialized StrategyOptimizer with {self.param_grid.total_combinations} combinations")

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate the base configuration.

        Args:
            config: Base configuration dictionary

        Raises:
            ValueError: If any required field is missing or invalid
        """
        required_fields = ['asset', 'data_path', 'initial_cash', 'base_currency']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")
                
        if not Path(config['data_path']).exists():
            raise ValueError(f"Data file not found: {config['data_path']}")
            
        if config['initial_cash'] <= 0:
            raise ValueError("Initial cash must be positive")

    def estimate_runtime(self, single_run_time: float = 60) -> Dict[str, Any]:
        """
        Estimate the total runtime for optimization.

        Args:
            single_run_time: Estimated time for a single backtest in seconds

        Returns:
            Dictionary containing runtime estimates
        """
        total_seconds = self.param_grid.estimate_calculation_time(single_run_time)
        
        return {
            'total_combinations': self.param_grid.total_combinations,
            'estimated_runtime': {
                'seconds': total_seconds,
                'minutes': total_seconds / 60,
                'hours': total_seconds / 3600
            },
            'parallel_estimate': {
                '4_workers': total_seconds / 4,
                '8_workers': total_seconds / 8,
                '16_workers': total_seconds / 16
            }
        }

    def run_optimization(self, test_run: bool = False) -> Dict[str, Any]:
        """
        Run the full optimization process.

        Args:
            test_run: If True, only run a small subset of combinations

        Returns:
            Dictionary containing optimization summary
        """
        try:
            start_time = time.time()
            logger.info("Starting optimization process")
            
            combinations = self.param_grid.generate_combinations()
            if test_run:
                combinations = combinations[:3]
                logger.info("Running in test mode with 3 combinations")
            
            optimization = create_optimization_tasks(
                param_combinations=combinations,
                base_config=self.base_config,
                output_dir=str(self.output_dir)
            )
            
            task_group = optimization.apply_async()
            monitor_task_progress(task_group)
            
            optimization_results = task_group.get()

            for result in optimization_results:
                if 'error' not in result:
                    self.results.add_result(params=result['parameters'], metrics=result['metrics'])
            
            self._generate_visualization_report()

            summary = self._create_optimization_summary(start_time, optimization_results)
            
            logger.info("Optimization completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info=True)
            raise

    def _generate_visualization_report(self) -> None:
        """Generate all visualization plots for the optimization results."""
        try:
            if self.results.results.empty:
                logger.warning("No results available for visualization")
                return
                
            logger.info("Generating visualization report")
            plots_dir = self.output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            self.visualizer.create_optimization_report()
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")

    def _create_optimization_summary(self, start_time: float, optimization_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a comprehensive optimization summary.

        Args:
            start_time: Time when optimization started
            optimization_results: List of results from backtests

        Returns:
            Dictionary summarizing the optimization process
        """
        try:
            summary = self.results.get_summary() if not self.results.results.empty else {}
            total_time = time.time() - start_time
            n_results = len(optimization_results)

            summary['runtime'] = {
                'total_seconds': total_time,
                'average_per_combination': total_time / n_results if n_results > 0 else 0
            }
            
            summary['parameters'] = self.param_grid.get_param_info()
            summary['file_locations'] = {
                'base_directory': str(self.output_dir),
                'plots_directory': str(self.output_dir / 'plots'),
                'results_csv': str(self.output_dir / 'optimization_results.csv'),
                'visualization_report': str(self.output_dir / 'plots' / 'optimization_report')
            }
            
            summary_path = self.output_dir / 'optimization_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info("Created optimization summary")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating optimization summary: {str(e)}")
            return {
                'runtime': {'total_seconds': 0, 'average_per_combination': 0},
                'parameters': {},
                'file_locations': {'base_directory': str(self.output_dir)}
            }
        
    def get_best_parameters(self, metric: str = 'total_return') -> Dict[str, Any]:
        """
        Get the best parameters based on a specific metric.

        Args:
            metric: Name of the metric to optimize for

        Returns:
            Dictionary containing the best parameters and their performance
        """
        try:
            return self.results.get_top_results(metric, n=1).to_dict('records')[0]
        except Exception as e:
            logger.error(f"Error getting best parameters: {str(e)}", exc_info=True)
            raise

    def save_results(self) -> None:
        """Save all results to CSV file."""
        try:
            results_path = self.output_dir / 'optimization_results.csv'
            self.results.save_to_csv(str(results_path))
            logger.info(f"Saved results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            raise
