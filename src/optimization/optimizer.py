"""
Strategy optimization coordinator module.

This module handles the main optimization process, including:
- Parameter configuration
- Task coordination
- Results management
- Visualization generation
- Report creation
"""

from __future__ import annotations

# Standard library imports
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, Final

# Local imports
from src.logger import setup_logger
from src.strategy.base_strategy import Strategy
from .task_coordinator import TaskCoordinator, OptimizationConfig
from .visualization.strategy_visualizer import StrategyVisualizer
from .visualization.report_generator import ReportGenerator

# Type aliases
PathLike = Union[str, Path]
ParamRanges = Dict[str, Dict[str, float]]
ConfigType = Dict[str, Any]
TimeEstimates = Dict[str, float]

# Constants
DEFAULT_TIMEOUT: Final[int] = 3600
DEFAULT_WORKERS: Final[int] = 4
DEFAULT_TIME_PER_RUN: Final[float] = 60.0

@dataclass(frozen=True)
class OptimizationResults:
    """Container for optimization results with immutable attributes."""
    summary: Dict[str, Any]
    best_params: Dict[str, Dict[str, float]]
    metrics: Dict[str, Dict[str, float]]
    execution_time: float
    plots_directory: Path
    report_path: Path

    def __post_init__(self) -> None:
        """Validate optimization results after initialization."""
        if not self.plots_directory.exists():
            raise ValueError(f"Plots directory does not exist: {self.plots_directory}")
        if not self.report_path.exists():
            raise ValueError(f"Report path does not exist: {self.report_path}")

class StrategyOptimizer:
    """Main optimizer class coordinating the complete optimization process."""
    
    def __init__(
        self, 
        strategy_class: Type[Strategy],
        param_ranges: ParamRanges,
        base_config: ConfigType,
        output_dir: Optional[PathLike] = None
    ) -> None:
        """
        Initialize the strategy optimizer.
        
        Args:
            strategy_class: Strategy class to optimize
            param_ranges: Parameter ranges for optimization
            base_config: Base configuration for backtesting
            output_dir: Directory for saving results
            
        Raises:
            ValueError: If parameters are invalid
            OSError: If output directory cannot be created
        """
        self._validate_inputs(strategy_class, param_ranges, base_config)
        
        self.strategy_class = strategy_class
        self.param_ranges = param_ranges
        self.base_config = base_config
        
        # Setup output directory
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir or f'optimization_results_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            'optimizer',
            str(self.output_dir / 'optimization.log')
        )
        
        self.logger.info(
            "Initializing optimizer for %s with %d parameters",
            strategy_class.__name__,
            len(param_ranges)
        )

    @staticmethod
    def _validate_inputs(
        strategy_class: Type[Strategy],
        param_ranges: ParamRanges,
        base_config: ConfigType
    ) -> None:
        """Validate initialization inputs."""
        if not issubclass(strategy_class, Strategy):
            raise ValueError("strategy_class must be a subclass of Strategy")
        
        if not param_ranges:
            raise ValueError("param_ranges cannot be empty")
            
        required_config = {'data_path', 'base_currency'}
        if not all(key in base_config for key in required_config):
            raise ValueError(f"base_config missing required keys: {required_config}")

    def run_optimization(
        self, 
        max_workers: int = DEFAULT_WORKERS,
        timeout: Optional[int] = None
    ) -> OptimizationResults:
        """
        Run the complete optimization process.
        
        Args:
            max_workers: Maximum number of parallel workers
            timeout: Optional timeout in seconds
            
        Returns:
            OptimizationResults containing results and metrics
            
        Raises:
            RuntimeError: If optimization fails
            TimeoutError: If optimization exceeds timeout
        """
        start_time = time.time()
        resources = {}
        
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
                task_timeout=timeout or DEFAULT_TIMEOUT
            )
            
            # Initialize components
            resources['coordinator'] = TaskCoordinator(config)
            optimization_results = resources['coordinator'].run_optimization()
            
            resources['visualizer'] = StrategyVisualizer(self.output_dir)
            plots = resources['visualizer'].generate_all_plots(
                optimization_results['results']
            )
            
            resources['report_generator'] = ReportGenerator(
                optimization_results,
                self.output_dir
            )
            report_path = resources['report_generator'].generate_markdown_report()
            
            # Compile results
            execution_time = time.time() - start_time
            results = OptimizationResults(
                summary=optimization_results['summary'],
                best_params=optimization_results['best_results'],
                metrics=optimization_results['metrics'],
                execution_time=execution_time,
                plots_directory=resources['visualizer'].plots_dir,
                report_path=report_path
            )
            
            self.logger.info(
                "Optimization completed in %.2f seconds. Results saved to %s",
                execution_time,
                self.output_dir
            )
            
            return results
            
        except Exception as e:
            self.logger.error("Error in optimization process: %s", str(e), exc_info=True)
            raise RuntimeError(f"Optimization failed: {str(e)}") from e
            
        finally:
            # Cleanup resources
            for resource in resources.values():
                if hasattr(resource, 'cleanup'):
                    try:
                        resource.cleanup()
                    except Exception as e:
                        self.logger.warning("Cleanup failed for %s: %s", resource, e)

    def estimate_runtime(
        self,
        time_per_run: float = DEFAULT_TIME_PER_RUN
    ) -> TimeEstimates:
        """
        Estimate total runtime for optimization.
        
        Args:
            time_per_run: Estimated time per parameter combination
            
        Returns:
            Dictionary containing runtime estimates
            
        Raises:
            ValueError: If parameters are invalid
        """
        if time_per_run <= 0:
            raise ValueError("time_per_run must be positive")
            
        try:
            total_combinations = self._calculate_combinations()
            total_runtime = total_combinations * time_per_run
            
            return {
                'total_combinations': total_combinations,
                'estimated_seconds': total_runtime,
                'estimated_minutes': total_runtime / 60,
                'estimated_hours': total_runtime / 3600
            }
            
        except Exception as e:
            self.logger.error("Error estimating runtime: %s", str(e), exc_info=True)
            raise

    def _calculate_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        total = 1
        for param_range in self.param_ranges.values():
            steps = (param_range['end'] - param_range['start']) / param_range['step']
            total *= (int(steps) + 1)
        return total

    @staticmethod
    def load_results(results_dir: PathLike) -> OptimizationResults:
        """
        Load optimization results from directory.
        
        Args:
            results_dir: Directory containing results
            
        Returns:
            OptimizationResults object
            
        Raises:
            FileNotFoundError: If required files are missing
            JSONDecodeError: If result files are invalid
        """
        results_dir = Path(results_dir)
        required_files = ['optimization_summary.json', 'optimization_results.json']
        
        # Verify required files exist
        for file in required_files:
            if not (results_dir / file).exists():
                raise FileNotFoundError(f"Missing required file: {file}")
        
        try:
            # Load files
            with open(results_dir / 'optimization_summary.json', 'r') as f:
                summary = json.load(f)
            
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
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid results file format: {e}") from e