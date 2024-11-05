"""Task coordinator for managing distributed optimization processes."""

import time
from typing import Dict, List, Any, Type
from dataclasses import dataclass
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

from src.optimization.worker_manager import WorkerManager
from src.optimization.celery_tasks import run_optimization
from src.strategy.base_strategy import Strategy
from src.optimization.parameter_grid import ParameterGrid
from src.logger import setup_logger

@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""
    strategy_class: Type[Strategy]
    param_ranges: Dict[str, Dict[str, float]]
    data_path: str
    base_config: Dict[str, Any]
    output_dir: Path
    max_parallel_tasks: int = 4
    task_timeout: int = 3600

class TaskCoordinator:
    """Coordinates optimization tasks and manages result collection."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize task coordinator.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = setup_logger('task_coordinator')
        self.worker_manager = WorkerManager(num_workers=config.max_parallel_tasks)
        self.param_grid = ParameterGrid(config.param_ranges)
        self.active_tasks: Dict[str, Any] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Any] = {}
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_optimization(self) -> Dict[str, Any]:
        """
        Run complete optimization process.
        
        Returns:
            Dict containing optimization results and statistics
        """
        try:
            start_time = time.time()
            self.logger.info("Starting optimization process")
            
            # Start worker infrastructure
            with self.worker_manager:
                # Generate parameter combinations
                combinations = self.param_grid.generate_combinations()
                total_combinations = len(combinations)
                self.logger.info(f"Generated {total_combinations} parameter combinations")
                
                # Submit tasks in batches
                completed = 0
                failed = 0
                results = []
                
                with ThreadPoolExecutor(max_workers=self.config.max_parallel_tasks) as executor:
                    futures = []
                    
                    for params in combinations:
                        future = executor.submit(
                            self._run_single_optimization,
                            params
                        )
                        futures.append(future)
                    
                    # Process results as they complete
                    for future in futures:
                        try:
                            result = future.result(timeout=self.config.task_timeout)
                            if 'error' in result:
                                failed += 1
                                self.failed_tasks[result['task_id']] = result
                            else:
                                completed += 1
                                self.completed_tasks[result['task_id']] = result
                                results.append(result)
                        except Exception as e:
                            failed += 1
                            self.logger.error(f"Task failed: {str(e)}")
                            
                        # Log progress
                        progress = (completed + failed) / total_combinations * 100
                        self.logger.info(f"Progress: {progress:.1f}% ({completed} completed, {failed} failed)")
                
            # Compile results
            execution_time = time.time() - start_time
            summary = self._compile_results(results, execution_time)
            
            # Save results
            self._save_results(results, summary)
            
            self.logger.info(f"Optimization completed in {execution_time:.2f} seconds")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in optimization process: {str(e)}")
            raise

    def _run_single_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run single optimization task."""
        try:
            # Submit task using Celery
            task = run_optimization.delay(
                strategy_class=self.config.strategy_class,
                params=params,
                data_path=self.config.data_path,
                config=self.config.base_config
            )
            
            self.active_tasks[task.id] = {
                'params': params,
                'start_time': time.time()
            }
            
            # Wait for result
            result = task.get(timeout=self.config.task_timeout)
            result['task_id'] = task.id
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in optimization task: {str(e)}")
            return {
                'task_id': task.id if 'task' in locals() else None,
                'error': str(e),
                'params': params
            }

    def _compile_results(self, results: List[Dict[str, Any]], execution_time: float) -> Dict[str, Any]:
        """Compile optimization results into summary."""
        try:
            summary = {
                'total_combinations': len(self.param_grid.generate_combinations()),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'execution_time': execution_time,
                'best_results': {},
                'parameter_statistics': {},
                'execution_statistics': {
                    'avg_task_time': sum(r['execution_time'] for r in results) / len(results) if results else 0,
                    'total_time': execution_time
                }
            }
            
            if results:
                # Find best results for each metric
                metrics = results[0]['metrics'].keys()
                for metric in metrics:
                    best_result = max(results, key=lambda x: x['metrics'][metric])
                    summary['best_results'][metric] = {
                        'value': best_result['metrics'][metric],
                        'parameters': best_result['parameters']
                    }
                
                # Calculate parameter statistics
                for param in self.config.param_ranges.keys():
                    param_values = [r['parameters'][param] for r in results]
                    summary['parameter_statistics'][param] = {
                        'min': min(param_values),
                        'max': max(param_values),
                        'mean': sum(param_values) / len(param_values)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error compiling results: {str(e)}")
            raise

    def _save_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        """Save optimization results and summary."""
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Save detailed results
            results_path = self.output_dir / f'optimization_results_{timestamp}.json'
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            # Save summary
            summary_path = self.output_dir / f'optimization_summary_{timestamp}.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            # Save failed tasks if any
            if self.failed_tasks:
                failed_path = self.output_dir / f'failed_tasks_{timestamp}.json'
                with open(failed_path, 'w', encoding='utf-8') as f:
                    json.dump(self.failed_tasks, f, indent=2)
            
            self.logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        if task_id in self.completed_tasks:
            return {'status': 'completed', **self.completed_tasks[task_id]}
        elif task_id in self.failed_tasks:
            return {'status': 'failed', **self.failed_tasks[task_id]}
        elif task_id in self.active_tasks:
            return {
                'status': 'running',
                'runtime': time.time() - self.active_tasks[task_id]['start_time'],
                'params': self.active_tasks[task_id]['params']
            }
        else:
            return {'status': 'unknown', 'task_id': task_id}

    def get_progress_report(self) -> Dict[str, Any]:
        """Get current optimization progress report."""
        total = len(self.param_grid.generate_combinations())
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        active = len(self.active_tasks)
        
        return {
            'total_combinations': total,
            'completed': completed,
            'failed': failed,
            'active': active,
            'progress_percentage': (completed + failed) / total * 100,
            'worker_status': self.worker_manager.get_worker_status()
        }