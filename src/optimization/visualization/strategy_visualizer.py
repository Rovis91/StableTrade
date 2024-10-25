import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from mpl_toolkits.mplot3d import Axes3D

from .base_visualizer import BaseVisualizer
from .data_validator import DataValidator

class StrategyVisualizer(BaseVisualizer):
    """Advanced visualization module for trading strategy optimization."""

    # Class constants
    METRICS = ['win_rate', 'sharpe_ratio', 'total_return']
    PARAMETERS = ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']
    
    def __init__(self, results_manager, output_dir: str, config=None):
        """Initialize the strategy visualizer.
        
        Args:
            results_manager: Results manager containing the optimization data
            output_dir: Directory for saving visualizations
            config: Optional visualization configuration
        """
        super().__init__(output_dir, config)
        self.results = results_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate data
        try:
            DataValidator.validate_results_data(self.results.results)
            self.logger.info("Data validation successful")
            self.logger.debug(f"Available columns: {list(self.results.results.columns)}")
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise

    def _create_plot_safely(self, plot_func: Callable, name: str) -> Optional[Path]:
        """Safely execute a plotting function with proper cleanup.
        
        Args:
            plot_func: Function that creates the plot
            name: Name of the plot for logging
            
        Returns:
            Optional[Path]: Path to saved plot if successful, None otherwise
        """
        try:
            plt.close('all')  # Clean up any existing plots
            return plot_func()
        except Exception as e:
            self.logger.error(f"Error creating {name} plot: {str(e)}")
            plt.close('all')  # Cleanup on error
            return None
         
    def create_performance_distribution(self) -> Path:
        """
        Generate performance distribution plot showing returns distribution.
        
        Returns:
            Path: Path to saved plot
        """
        try:
            fig, ax = plt.subplots()
            
            returns = self.results.results['total_return']
            
            # Main distribution plot
            sns.histplot(returns, kde=True, ax=ax)
            
            if self.config.show_annotations:
                self._add_distribution_annotations(ax, returns)
            
            ax.set_title('Distribution des Rendements Totaux')
            ax.set_xlabel('Rendement Total (%)')
            ax.set_ylabel('Fréquence')
            
            return self.save_plot('returns_distribution')
            
        except Exception as e:
            self.logger.error(f"Error creating performance distribution: {str(e)}")
            raise

    def _add_distribution_annotations(self, ax: plt.Axes, returns: pd.Series) -> None:
        """Add statistical annotations to the distribution plot."""
        # Add percentile lines
        percentiles = [25, 50, 75]
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        for p, c in zip(percentiles, colors):
            val = np.percentile(returns, p)
            ax.axvline(val, color=c, linestyle='--',
                      label=f'{p}th percentile: {val:.2f}%')
        
        # Add statistical moments
        mean = returns.mean()
        std = returns.std()
        skew = returns.skew()
        
        stats_text = (
            f'Mean: {mean:.2f}%\n'
            f'Std: {std:.2f}%\n'
            f'Skew: {skew:.2f}'
        )
        
        # Add text box with statistics
        ax.text(0.95, 0.95, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper left')

    def create_risk_scatter(self) -> Path:
        """
        Generate risk-return scatter plot.
        
        Returns:
            Path: Path to saved plot
        """
        try:
            fig, ax = plt.subplots()
            
            scatter = ax.scatter(
                self.results.results['max_drawdown'],
                self.results.results['total_return'],
                c=self.results.results['sharpe_ratio'],
                s=self.results.results['win_rate'] * 5,
                alpha=0.6,
                cmap='viridis'
            )
            
            if self.config.show_annotations:
                self._add_scatter_annotations(ax)
            
            plt.colorbar(scatter, label='Ratio de Sharpe')
            
            ax.set_xlabel('Drawdown Maximum (%)')
            ax.set_ylabel('Rendement Total (%)')
            ax.set_title('Analyse Risque-Rendement')
            
            return self.save_plot('risk_return_scatter')
            
        except Exception as e:
            self.logger.error(f"Error creating risk scatter: {str(e)}")
            raise

    def _save_visualization_config(self, generated_plots: Dict[str, Path]) -> None:
        """Save visualization configuration to JSON."""
        try:
            config_path = self.plots_dir / 'visualization_config.json'
            config_data = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config.to_dict(),
                'plots': {k: str(v) for k, v in generated_plots.items()}
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            self.logger.info(f"Visualization config saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving visualization config: {str(e)}")

    def _add_scatter_annotations(self, ax: plt.Axes) -> None:
        """Add annotations to the risk scatter plot."""
        # Find best performing point
        best_idx = self.results.results['sharpe_ratio'].idxmax()
        best_point = self.results.results.loc[best_idx]
        
        # Add annotation
        ax.annotate('Optimal',
                   (best_point['max_drawdown'], best_point['total_return']),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->'))

    def create_robustness_violin(self) -> Path:
        """Generate violin plot for strategy robustness analysis."""
        try:
            metrics = ['win_rate', 'sharpe_ratio', 'total_return']
            data = []
            
            for param in ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']:
                for metric in metrics:
                    param_values = self.results.results[param].unique()
                    for val in param_values:
                        metric_values = self.results.results[
                            self.results.results[param] == val][metric]
                        data.extend([{
                            'Parameter': f'{param}\n({val})',
                            'Metric': metric,
                            'Value': v
                        } for v in metric_values])
            
            plt.figure(figsize=(15, 10))
            df = pd.DataFrame(data)
            sns.violinplot(data=df, x='Parameter', y='Value', hue='Metric')
            plt.xticks(rotation=45)
            plt.title('Analyse de Robustesse des Paramètres', fontsize=14)
            
            return self.save_plot('robustness_violin')
            
        except Exception as e:
            self.logger.error(f"Error creating robustness violin plot: {str(e)}")
            raise

    def create_3d_surface(self) -> Path:
        """Generate 3D surface plot for parameter optimization."""
        try:
            pivot = self.results.results.pivot_table(
                values='sharpe_ratio',
                index='depeg_threshold',
                columns='trade_amount',
                aggfunc='mean'
            )
            
            X, Y = np.meshgrid(pivot.columns, pivot.index)
            Z = pivot.values
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(X, Y, Z, cmap='viridis')
            
            ax.set_xlabel('Trade Amount')
            ax.set_ylabel('Depeg Threshold')
            ax.set_zlabel('Sharpe Ratio')
            
            plt.colorbar(surf)
            plt.title("Surface d'Optimisation: Impact des Paramètres sur le Sharpe Ratio")
            
            return self.save_plot('parameter_surface_3d')
            
        except Exception as e:
            self.logger.error(f"Error creating 3D surface plot: {str(e)}")
            raise

    def create_trade_duration_analysis(self) -> Path:
        """
        Generate trade analysis plots using available metrics.
        
        Returns:
            Path: Path to saved plot
        """
        try:
            self.logger.info(f"Available columns: {list(self.results.results.columns)}")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
            
            # Plot 1: Distribution of returns by trade amount
            sns.boxplot(
                x='trade_amount',
                y='total_return',
                data=self.results.results,
                ax=ax1
            )
            ax1.set_title('Distribution des Rendements par Taille de Trade')
            ax1.set_xlabel('Taille du Trade')
            ax1.set_ylabel('Rendement Total (%)')
            ax1.tick_params(axis='x', rotation=45)

            # Plot 2: Win rate analysis
            avg_win_rates = self.results.results.groupby('trade_amount')['win_rate'].mean()
            ax2.bar(
                avg_win_rates.index.astype(str),
                avg_win_rates.values,
                alpha=0.75
            )
            ax2.set_title('Taux de Réussite Moyen par Taille de Trade')
            ax2.set_xlabel('Taille du Trade')
            ax2.set_ylabel('Taux de Réussite (%)')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return self.save_plot('trade_analysis')
            
        except Exception as e:
            self.logger.error(f"Error creating trade duration analysis: {str(e)}")
            raise

    def generate_complete_analysis(self) -> Dict[str, Path]:
        """Generate all visualization components."""
        try:
            self.logger.info("Starting complete analysis generation...")
            generated_plots = {}
            
            # Define plots to generate
            plots_to_generate = [
                ('performance', self.create_performance_distribution),
                ('risk_scatter', self.create_risk_scatter),
                ('robustness', self.create_robustness_violin),
                ('surface', self.create_3d_surface),
                ('trades', self.create_trade_duration_analysis)
            ]

            # Generate plots safely
            for plot_name, plot_func in plots_to_generate:
                plot_path = self._create_plot_safely(plot_func, plot_name)
                if plot_path:
                    generated_plots[plot_name] = plot_path
                    self.logger.info(f"{plot_name} plot generated successfully")

            # Save configuration
            self._save_visualization_config(generated_plots)
            
            return generated_plots
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis generation: {str(e)}")
            raise

    @property
    def available_metrics(self) -> List[str]:
        """Get list of available metrics in the results."""
        return [col for col in self.results.results.columns 
                if col in self.METRICS]

    @property
    def available_parameters(self) -> List[str]:
        """Get list of available parameters in the results."""
        return [col for col in self.results.results.columns 
                if col in self.PARAMETERS]
    