import json
import logging
import time
from typing import Dict, Any, Type, Optional
from pathlib import Path
from dataclasses import dataclass
from src.logger import setup_logger
from .task_coordinator import TaskCoordinator, OptimizationConfig
from src.strategy.base_strategy import Strategy
from src.optimization.visualization.strategy_visualizer import StrategyVisualizer
from src.optimization.visualization.report_generator import ReportGenerator


def _setup_logging(self) -> logging.Logger:
    """Setup logging configuration."""
    log_file = self.output_dir / 'optimization.log'
    return setup_logger('optimizer', str(log_file))

@dataclass
class OptimizationResults:
    """Container for optimization results."""
    summary: Dict[str, Any]
    best_params: Dict[str, Dict[str, float]]
    metrics: Dict[str, Dict[str, float]]
    execution_time: float
    plots_directory: Path
    report_path: Path

class StrategyOptimizer:
    """Main optimizer class coordinating the optimization process."""
    
    def __init__(self, 
                 strategy_class: Type[Strategy],
                 param_ranges: Dict[str, Dict[str, float]],
                 base_config: Dict[str, Any],
                 output_dir: Optional[str] = None):
        """
        Initialize the strategy optimizer.
        
        Args:
            strategy_class: The strategy class to optimize
            param_ranges: Parameter ranges for optimization
            base_config: Base configuration for backtesting
            output_dir: Directory for saving results
        """
        self.strategy_class = strategy_class
        self.param_ranges = param_ranges
        self.base_config = base_config
        
        # Setup output directory
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir or f'optimization_results_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        self.logger.info(
            f"Initializing optimizer for {strategy_class.__name__} "
            f"with {len(param_ranges)} parameters"
        )

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        fh = logging.FileHandler(self.output_dir / 'optimization.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def run_optimization(self, 
                        max_workers: int = 4,
                        timeout: Optional[int] = None) -> OptimizationResults:
        """
        Run the complete optimization process.
        
        Args:
            max_workers: Maximum number of parallel workers
            timeout: Optional timeout in seconds
            
        Returns:
            OptimizationResults containing all optimization results
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting optimization process")
            
            # Create optimization config
            config = OptimizationConfig(
                strategy_class=self.strategy_class,
                param_ranges=self.param_ranges,
                data_path=self.base_config['data_path'],
                base_config=self.base_config,
                output_dir=self.output_dir,
                max_parallel_tasks=max_workers,
                task_timeout=timeout or 3600
            )
            
            # Run optimization
            coordinator = TaskCoordinator(config)
            optimization_results = coordinator.run_optimization()
            
            # Generate visualizations
            visualizer = StrategyVisualizer(self.output_dir)
            plots = visualizer.generate_all_plots(optimization_results['results'])
            
            # Generate report
            report_generator = ReportGenerator(optimization_results, self.output_dir)
            report_path = report_generator.generate_markdown_report()
            
            # Compile results
            execution_time = time.time() - start_time
            results = OptimizationResults(
                summary=optimization_results['summary'],
                best_params=optimization_results['best_results'],
                metrics=optimization_results['metrics'],
                execution_time=execution_time,
                plots_directory=visualizer.plots_dir,
                report_path=report_path
            )
            
            self.logger.info(
                f"Optimization completed in {execution_time:.2f} seconds. "
                f"Results saved to {self.output_dir}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in optimization process: {str(e)}")
            raise

    def estimate_runtime(self, time_per_run: float = 60.0) -> Dict[str, float]:
        """
        Estimate total runtime for optimization.
        
        Args:
            time_per_run: Estimated time per parameter combination in seconds
            
        Returns:
            Dictionary containing runtime estimates
        """
        try:
            # Calculate total combinations
            total_combinations = 1
            for param_range in self.param_ranges.values():
                steps = (param_range['end'] - param_range['start']) / param_range['step']
                total_combinations *= (int(steps) + 1)
            
            # Calculate estimates
            total_runtime = total_combinations * time_per_run
            
            return {
                'total_combinations': total_combinations,
                'estimated_seconds': total_runtime,
                'estimated_minutes': total_runtime / 60,
                'estimated_hours': total_runtime / 3600
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating runtime: {str(e)}")
            raise

    @staticmethod
    def load_results(results_dir: str) -> OptimizationResults:
        """
        Load optimization results from directory.
        
        Args:
            results_dir: Directory containing optimization results
            
        Returns:
            OptimizationResults object
        """
        results_dir = Path(results_dir)
        
        # Load summary
        with open(results_dir / 'optimization_summary.json', 'r') as f:
            summary = json.load(f)
        
        # Load results
        with open(results_dir / 'optimization_results.json', 'r') as f:
            results = json.load(f)
        
        return OptimizationResults(
            summary=summary,
            best_params=summary['best_results'],
            metrics=summary.get('metrics', {}),
            execution_time=summary.get('execution_time', 0),
            plots_directory=results_dir / 'plots',
            report_path=results_dir / 'optimization_report.md'
        )