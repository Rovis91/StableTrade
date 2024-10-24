import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class OptimizationResults:
    """
    Manages and analyzes optimization results, including the storage, 
    analysis, and export of parameter combinations and performance metrics.
    """
    
    def __init__(self):
        """Initialize the OptimizationResults manager."""
        self.results = pd.DataFrame()
        self.param_columns: List[str] = []
        self.metric_columns: List[str] = []
        logger.info("Initialized OptimizationResults manager")

    def add_result(self, params: Dict[str, float], metrics: Dict[str, float]) -> None:
        """
        Add a single optimization result.
        
        Args:
            params (Dict[str, float]): Dictionary of parameter names and values.
            metrics (Dict[str, float]): Dictionary of metric names and values.
        """
        if not self.validate_result(params, metrics):
            raise ValueError("Invalid result format")
        
        try:
            row = {**params, **metrics}
            self.results = pd.concat([self.results, pd.DataFrame([row])], ignore_index=True)

            if not self.param_columns:
                self.param_columns = list(params.keys())
                self.metric_columns = list(metrics.keys())
                
            logger.debug(f"Added result: params={params}, metrics={metrics}")
        except Exception as e:
            logger.error(f"Error adding result: {str(e)}")
            raise

    def validate_result(self, params: Dict[str, float], metrics: Dict[str, float]) -> bool:
        """
        Validate new results before adding them.
        
        Args:
            params (Dict[str, float]): Dictionary of parameter values.
            metrics (Dict[str, float]): Dictionary of metric values.
            
        Returns:
            bool: True if the result is valid, False otherwise.
        """
        if not self.param_columns:  # First result defines the structure
            return all(isinstance(v, (int, float)) for v in {**params, **metrics}.values())
        
        return (set(params.keys()) == set(self.param_columns) and 
                set(metrics.keys()) == set(self.metric_columns) and
                all(isinstance(v, (int, float)) for v in {**params, **metrics}.values()))

    def save_to_csv(self, filepath: str) -> None:
        """
        Save results to a CSV file.
        
        Args:
            filepath (str): Path where the CSV file will be saved.
        """
        try:
            if self.results.empty:
                logger.warning("No results to save")
                return

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self.results.to_csv(filepath, index=False)
            logger.info(f"Saved {len(self.results)} results to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {str(e)}")
            raise

    def load_from_csv(self, filepath: str) -> None:
        """
        Load results from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
        """
        try:
            self.results = pd.read_csv(filepath)
            self._update_column_lists()
            logger.info(f"Loaded {len(self.results)} results from {filepath}")
        except Exception as e:
            logger.error(f"Error loading results from CSV: {str(e)}")
            raise

    def _update_column_lists(self) -> None:
        """Update param_columns and metric_columns after loading data."""
        if not self.results.empty:
            all_columns = set(self.results.columns)
            self.param_columns = [col for col in all_columns 
                                  if any(param in col.lower() for param in ['threshold', 'amount', 'stop', 'profit'])]
            self.metric_columns = list(all_columns - set(self.param_columns))

    def filter_results(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter results based on conditions.
        
        Args:
            conditions (Dict[str, Any]): Dict of column names and their filter conditions.
                e.g., {'total_return': lambda x: x > 0, 'sharpe_ratio': lambda x: x > 1}
        
        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        filtered = self.results.copy()
        for col, condition in conditions.items():
            filtered = filtered[filtered[col].apply(condition)]
        return filtered

    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for parameters and metrics.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing statistics for each parameter and metric.
        """
        if self.results.empty:
            logger.warning("No results available for statistics calculation")
            return {}

        stats = {}
        try:
            for col in self.param_columns + self.metric_columns:
                if not self.results[col].empty:
                    stats[col] = {
                        'min': float(self.results[col].min()),
                        'max': float(self.results[col].max()),
                        'mean': float(self.results[col].mean()),
                        'std': float(self.results[col].std()),
                        'median': float(self.results[col].median())
                    }
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}
        return stats

    def calculate_correlations(self) -> pd.DataFrame:
        """
        Calculate correlations between parameters and metrics.
        
        Returns:
            pd.DataFrame: DataFrame containing the correlation matrix.
        """
        if self.results.empty:
            logger.warning("No results available for correlation calculation")
            return pd.DataFrame()

        try:
            correlations = self.results[self.param_columns + self.metric_columns].corr()
            logger.info("Calculated correlation matrix")
            return correlations
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            return pd.DataFrame()

    def get_parameter_performance(self, param_name: str, metric_name: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific parameter.
        
        Args:
            param_name (str): Name of the parameter to analyze.
            metric_name (str): Name of the metric to use for analysis.
            
        Returns:
            Dict[str, Any]: Dictionary containing parameter performance analysis.
        """
        if param_name not in self.param_columns or metric_name not in self.metric_columns:
            raise ValueError(f"Invalid parameter '{param_name}' or metric '{metric_name}'")

        try:
            grouped = self.results.groupby(param_name)[metric_name].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).round(6)
            
            best_value = self.results.loc[self.results[metric_name].idxmax(), param_name]
            worst_value = self.results.loc[self.results[metric_name].idxmin(), param_name]
            
            analysis = {
                'value_stats': grouped.to_dict('index'),
                'best_value': float(best_value),
                'worst_value': float(worst_value),
                'unique_values': sorted(self.results[param_name].unique().tolist())
            }
            
            logger.info(f"Analyzed performance for parameter '{param_name}' using metric '{metric_name}'")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing parameter performance: {str(e)}")
            raise

    def analyze_parameter_sensitivity(self, param_name: str, metric_name: str) -> Dict[str, Any]:
        """
        Analyze how changes in a parameter affect a specific metric.
        
        Args:
            param_name (str): Name of the parameter to analyze.
            metric_name (str): Name of the metric to analyze.
            
        Returns:
            Dict[str, Any]: Dictionary containing sensitivity analysis.
        """
        if self.results.empty:
            return {}
            
        analysis = {}
        param_values = sorted(self.results[param_name].unique())
        
        sensitivity = []
        for i in range(len(param_values) - 1):
            curr_value, next_value = param_values[i], param_values[i + 1]
            
            curr_metric = self.results[self.results[param_name] == curr_value][metric_name].mean()
            next_metric = self.results[self.results[param_name] == next_value][metric_name].mean()
            
            param_change = next_value - curr_value
            metric_change = next_metric - curr_metric
            sensitivity.append({
                'param_range': (curr_value, next_value),
                'metric_change': metric_change,
                'sensitivity': metric_change / param_change if param_change != 0 else 0
            })
        
        analysis['sensitivity_by_range'] = sensitivity
        return analysis

    def get_top_results(self, metric_name: str, n: int = 10) -> pd.DataFrame:
        """
        Get the top N parameter combinations based on a specific metric.
        
        Args:
            metric_name (str): Name of the metric to sort by.
            n (int): Number of top results to return (default: 10).
            
        Returns:
            pd.DataFrame: DataFrame containing the top N results.
        """
        if metric_name not in self.metric_columns:
            raise ValueError(f"Invalid metric name: {metric_name}")

        try:
            top_results = self.results.nlargest(n, metric_name)
            logger.info(f"Retrieved top {n} results by {metric_name}")
            return top_results
        except Exception as e:
            logger.error(f"Error getting top results: {str(e)}")
            raise

    def compare_results(self, other: 'OptimizationResults', metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare results with another optimization run.
        
        Args:
            other (OptimizationResults): Another OptimizationResults instance to compare with.
            metrics (Optional[List[str]]): List of metrics to compare (optional).
            
        Returns:
            Dict[str, Any]: Dictionary containing comparison statistics.
        """
        if metrics is None:
            metrics = self.metric_columns
            
        comparison = {}
        for metric in metrics:
            comparison[metric] = {
                'this_mean': float(self.results[metric].mean()),
                'other_mean': float(other.results[metric].mean()),
                'difference_pct': float(((self.results[metric].mean() / 
                                      other.results[metric].mean()) - 1) * 100),
                'this_best': float(self.results[metric].max()),
                'other_best': float(other.results[metric].max())
            }
        return comparison

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization results.
        
        Returns:
            Dict[str, Any]: Dictionary containing summary information.
        """
        if self.results.empty:
            logger.warning("No results available for summary")
            return {}

        try:
            summary = {
                'total_combinations': len(self.results),
                'parameters_tested': self.param_columns,
                'metrics_evaluated': self.metric_columns,
                'best_results': {
                    metric: {
                        'value': float(self.results[metric].max()),
                        'params': self.results.loc[self.results[metric].idxmax(), self.param_columns].to_dict()
                    }
                    for metric in self.metric_columns
                },
                'parameter_ranges': {
                    param: {
                        'min': float(self.results[param].min()),
                        'max': float(self.results[param].max()),
                        'unique_values': len(self.results[param].unique())
                    }
                    for param in self.param_columns
                }
            }
            
            logger.info("Generated optimization results summary")
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {}
