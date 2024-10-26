"""Enhanced visualization module for strategy optimization with improved plots."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path

class StrategyVisualizer:
    """Enhanced visualization class with improved plotting capabilities."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Set style configurations
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl", 8)
        
        # Custom color maps
        self.returns_cmap = LinearSegmentedColormap.from_list(
            'returns', ['#FF9B9B', '#FFEC9B', '#9BFFB8']
        )
        self.drawdown_cmap = LinearSegmentedColormap.from_list(
            'drawdown', ['#9BFFB8', '#FFEC9B', '#FF9B9B']
        )

    def create_performance_surface(self, 
                                 df: pd.DataFrame,
                                 param1: str,
                                 param2: str,
                                 metric: str = 'sharpe_ratio',
                                 title: Optional[str] = None) -> Path:
        """
        Create an enhanced 3D surface plot for parameter optimization.
        
        Args:
            df: DataFrame with optimization results
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to plot (default: 'sharpe_ratio')
            title: Optional plot title
            
        Returns:
            Path: Path to saved plot
        """
        try:
            # Create pivot table for surface
            pivot = df.pivot_table(
                values=metric,
                index=param1,
                columns=param2,
                aggfunc='mean'
            )
            
            # Create meshgrid
            X, Y = np.meshgrid(pivot.columns, pivot.index)
            Z = pivot.values
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 2, height_ratios=[3, 1])
            
            # 3D Surface plot
            ax1 = fig.add_subplot(gs[0, :], projection='3d')
            surf = ax1.plot_surface(
                X, Y, Z,
                cmap='viridis',
                linewidth=0,
                antialiased=True,
                alpha=0.8
            )
            
            # Customize 3D plot
            ax1.set_xlabel(param2)
            ax1.set_ylabel(param1)
            ax1.set_zlabel(metric)
            if title:
                ax1.set_title(title)
                
            # Add color bar
            fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
            
            # Add contour plot
            ax2 = fig.add_subplot(gs[1, 0])
            contour = ax2.contour(X, Y, Z, cmap='viridis', levels=10)
            ax2.clabel(contour, inline=True, fontsize=8)
            ax2.set_xlabel(param2)
            ax2.set_ylabel(param1)
            ax2.set_title('Contour View')
            
            # Add heatmap
            ax3 = fig.add_subplot(gs[1, 1])
            sns.heatmap(
                pivot,
                cmap='viridis',
                ax=ax3,
                cbar_kws={'label': metric}
            )
            ax3.set_title('Heatmap View')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / f'surface_{param1}_{param2}_{metric}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating performance surface: {str(e)}")
            raise

    def create_distribution_analysis(self, df: pd.DataFrame) -> Path:
        """
        Create enhanced distribution analysis plots.
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            Path: Path to saved plot
        """
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 2)
            
            # Returns distribution
            ax1 = fig.add_subplot(gs[0, 0])
            sns.histplot(
                data=df,
                x='total_return',
                kde=True,
                ax=ax1,
                color='blue',
                alpha=0.3
            )
            ax1.axvline(
                df['total_return'].mean(),
                color='red',
                linestyle='--',
                label=f'Mean: {df["total_return"].mean():.2f}%'
            )
            ax1.set_title('Distribution des Rendements')
            ax1.set_xlabel('Rendement Total (%)')
            ax1.legend()
            
            # Sharpe ratio distribution
            ax2 = fig.add_subplot(gs[0, 1])
            sns.histplot(
                data=df,
                x='sharpe_ratio',
                kde=True,
                ax=ax2,
                color='green',
                alpha=0.3
            )
            ax2.axvline(
                df['sharpe_ratio'].mean(),
                color='red',
                linestyle='--',
                label=f'Mean: {df["sharpe_ratio"].mean():.2f}'
            )
            ax2.set_title('Distribution des Ratios de Sharpe')
            ax2.set_xlabel('Ratio de Sharpe')
            ax2.legend()
            
            # Drawdown distribution
            ax3 = fig.add_subplot(gs[1, 0])
            sns.histplot(
                data=df,
                x='max_drawdown',
                kde=True,
                ax=ax3,
                color='red',
                alpha=0.3
            )
            ax3.axvline(
                df['max_drawdown'].mean(),
                color='blue',
                linestyle='--',
                label=f'Mean: {df["max_drawdown"].mean():.2f}%'
            )
            ax3.set_title('Distribution des Drawdowns Maximum')
            ax3.set_xlabel('Drawdown Maximum (%)')
            ax3.legend()
            
            # Win rate distribution
            ax4 = fig.add_subplot(gs[1, 1])
            sns.histplot(
                data=df,
                x='win_rate',
                kde=True,
                ax=ax4,
                color='purple',
                alpha=0.3
            )
            ax4.axvline(
                df['win_rate'].mean(),
                color='red',
                linestyle='--',
                label=f'Mean: {df["win_rate"].mean():.2f}%'
            )
            ax4.set_title('Distribution des Taux de Réussite')
            ax4.set_xlabel('Taux de Réussite (%)')
            ax4.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / 'distribution_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating distribution analysis: {str(e)}")
            raise

    def create_correlation_matrix(self, df: pd.DataFrame) -> Path:
        """
        Create enhanced correlation matrix visualization.
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            Path: Path to saved plot
        """
        try:
            # Select relevant columns
            params = ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']
            metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
            
            # Calculate correlations
            corr = df[params + metrics].corr()
            
            # Create plot
            plt.figure(figsize=(12, 10))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr))
            
            # Create heatmap
            sns.heatmap(
                corr,
                mask=mask,
                cmap='RdYlBu_r',
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'}
            )
            
            plt.title('Matrice de Corrélation des Paramètres et Métriques')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / 'correlation_matrix.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {str(e)}")
            raise

    def create_parameter_impact(self, df: pd.DataFrame) -> Path:
        """
        Create parameter impact visualization.
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            Path: Path to saved plot
        """
        try:
            params = ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']
            metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
            
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(len(params), 1, height_ratios=[1] * len(params))
            
            for i, param in enumerate(params):
                ax = fig.add_subplot(gs[i])
                
                # Create violin plots for each metric
                data = []
                for metric in metrics:
                    for val in df[param].unique():
                        data.append({
                            'Parameter': f'{param} ({val:.3f})',
                            'Metric': metric,
                            'Value': df[df[param] == val][metric].mean()
                        })
                
                plot_df = pd.DataFrame(data)
                sns.violinplot(
                    data=plot_df,
                    x='Parameter',
                    y='Value',
                    hue='Metric',
                    ax=ax,
                    cut=0
                )
                
                ax.set_title(f'Impact de {param}')
                ax.tick_params(axis='x', rotation=45)
                
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / 'parameter_impact.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating parameter impact visualization: {str(e)}")
            raise

    def generate_all_plots(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Generate all visualization plots.
        
        Args:
            df: DataFrame with optimization results
            
        Returns:
            Dict[str, Path]: Dictionary mapping plot names to file paths
        """
        try:
            plots = {}
            
            # Generate surface plots for each parameter combination
            params = ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']
            for i, param1 in enumerate(params[:-1]):
                for param2 in params[i+1:]:
                    plot_path = self.create_performance_surface(
                        df,
                        param1,
                        param2,
                        'sharpe_ratio',
                        f'Impact de {param1} et {param2} sur le Ratio de Sharpe'
                    )
                    plots[f'surface_{param1}_{param2}'] = plot_path
            
            # Generate distribution analysis
            plots['distributions'] = self.create_distribution_analysis(df)
            
            # Generate correlation matrix
            plots['correlations'] = self.create_correlation_matrix(df)
            
            # Generate parameter impact
            plots['parameter_impact'] = self.create_parameter_impact(df)
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            raise