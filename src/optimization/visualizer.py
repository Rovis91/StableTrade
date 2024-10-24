import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class OptimizationVisualizer:
    """
    Creates visualizations for optimization results.

    Handles the creation of various plots for analyzing optimization results,
    including parameter performance plots, correlation heatmaps, and metric
    distribution charts.
    """
    
    def __init__(self, results_manager, output_dir: str = 'optimization_plots'):
        """
        Initialize the visualization module.

        Args:
            results_manager: OptimizationResults instance containing the results.
            output_dir (str): Directory where plots will be saved.
        """
        self.results = results_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn-v0_8')
        self.default_figsize = (10, 6)
        self.default_dpi = 100
        
        logger.info(f"Initialized OptimizationVisualizer with output directory: {output_dir}")

    def plot_parameter_performance(self, param_name: str, metric_name: str,
                                   show_error_bars: bool = True) -> str:
        """
        Create a bar plot showing parameter performance with error bars.

        Args:
            param_name (str): Name of the parameter to plot.
            metric_name (str): Name of the metric to analyze.
            show_error_bars (bool): Whether to show error bars on the plot.
            
        Returns:
            str: Path to the saved plot file.
        """
        try:
            performance = self.results.get_parameter_performance(param_name, metric_name)
            
            plt.figure(figsize=self.default_figsize, dpi=self.default_dpi)
            
            values = list(performance['value_stats'].keys())
            means = [stats['mean'] for stats in performance['value_stats'].values()]
            stds = [stats['std'] for stats in performance['value_stats'].values()]
            
            plt.bar(values, means, yerr=stds if show_error_bars else None, capsize=5, alpha=0.6)
            
            plt.title(f'Performance Analysis: {param_name} vs {metric_name}')
            plt.xlabel(param_name)
            plt.ylabel(metric_name)
            plt.grid(True, alpha=0.3)
            
            plt.axvline(x=performance['best_value'], color='g', linestyle='--', alpha=0.5, label='Best Value')
            plt.axvline(x=performance['worst_value'], color='r', linestyle='--', alpha=0.5, label='Worst Value')
            plt.legend()
            
            output_path = self.output_dir / f'param_performance_{param_name}_{metric_name}.png'
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Created parameter performance plot: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating parameter performance plot: {str(e)}")
            raise

    def plot_correlation_heatmap(self) -> str:
        """
        Create a correlation heatmap for parameters and metrics.

        Returns:
            str: Path to the saved plot file.
        """
        try:
            correlations = self.results.calculate_correlations()
            
            plt.figure(figsize=(12, 8), dpi=self.default_dpi)
            
            sns.heatmap(correlations, annot=True, cmap='RdBu', center=0, fmt='.2f', square=True)
            
            plt.title('Parameter and Metric Correlations')
            
            output_path = self.output_dir / 'correlation_heatmap.png'
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created correlation heatmap: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            raise

    def plot_metric_distributions(self, metrics: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create distribution plots for specified metrics.

        Args:
            metrics (Optional[List[str]]): List of metrics to plot (optional, plots all if None).
            
        Returns:
            Dict[str, str]: Dictionary mapping metric names to plot file paths.
        """
        try:
            if metrics is None:
                metrics = self.results.metric_columns
                
            plot_paths = {}
            
            for metric in metrics:
                plt.figure(figsize=self.default_figsize, dpi=self.default_dpi)
                
                sns.histplot(self.results.results[metric], kde=True)
                
                plt.title(f'Distribution of {metric}')
                plt.xlabel(metric)
                plt.ylabel('Count')
                plt.grid(True, alpha=0.3)
                
                mean_value = self.results.results[metric].mean()
                median_value = self.results.results[metric].median()
                
                plt.axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')
                plt.axvline(median_value, color='g', linestyle='--', label=f'Median: {median_value:.2f}')
                plt.legend()
                
                output_path = self.output_dir / f'metric_distribution_{metric}.png'
                plt.savefig(output_path)
                plt.close()
                
                plot_paths[metric] = str(output_path)
                
            logger.info(f"Created distribution plots for {len(metrics)} metrics")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating metric distribution plots: {str(e)}")
            raise

    def plot_parameter_sensitivity(self, param_name: str, metric_name: str) -> str:
        """
        Create a sensitivity analysis plot for a parameter.

        Args:
            param_name (str): Name of the parameter to analyze.
            metric_name (str): Name of the metric to analyze.
            
        Returns:
            str: Path to the saved plot file.
        """
        try:
            sensitivity = self.results.analyze_parameter_sensitivity(param_name, metric_name)
            
            plt.figure(figsize=self.default_figsize, dpi=self.default_dpi)
            
            param_ranges = [f"{r['param_range'][0]}-{r['param_range'][1]}" 
                            for r in sensitivity['sensitivity_by_range']]
            sensitivities = [r['sensitivity'] for r in sensitivity['sensitivity_by_range']]
            
            plt.bar(param_ranges, sensitivities, alpha=0.6)
            
            plt.title(f'Parameter Sensitivity: {param_name} vs {metric_name}')
            plt.xlabel(f'{param_name} Ranges')
            plt.ylabel(f'Sensitivity (Δ{metric_name}/Δ{param_name})')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            output_path = self.output_dir / f'param_sensitivity_{param_name}_{metric_name}.png'
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created parameter sensitivity plot: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating parameter sensitivity plot: {str(e)}")
            raise

    def create_optimization_report(self) -> str:
        """
        Create a comprehensive visualization report with all plots.

        Returns:
            str: Path to the report directory containing all plots.
        """
        try:
            logger.info("Starting optimization report generation")
            
            report_dir = self.output_dir / 'optimization_report'
            report_dir.mkdir(exist_ok=True)
            
            self.plot_correlation_heatmap()
            
            for param in self.results.param_columns:
                for metric in self.results.metric_columns:
                    self.plot_parameter_performance(param, metric)
                    self.plot_parameter_sensitivity(param, metric)
            
            self.plot_metric_distributions()
            
            logger.info(f"Generated optimization report in {report_dir}")
            return str(report_dir)
            
        except Exception as e:
            logger.error(f"Error creating optimization report: {str(e)}")
            raise

    def set_style(self, style: str = 'seaborn', figsize: tuple = (10, 6),
                  dpi: int = 100) -> None:
        """
        Set the plotting style.

        Args:
            style (str): Name of the matplotlib style to use.
            figsize (tuple): Default figure size (width, height).
            dpi (int): Default DPI for plots.
        """
        plt.style.use(style)
        self.default_figsize = figsize
        self.default_dpi = dpi
        logger.info(f"Updated plotting style: {style}, figsize: {figsize}, dpi: {dpi}")
