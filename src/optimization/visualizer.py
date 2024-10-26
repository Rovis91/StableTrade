from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from datetime import datetime

@dataclass
class VisualizationConfig:
    """Configuration settings for visualizations."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = 'seaborn-v0_8-darkgrid'
    color_palette: str = "husl"
    n_colors: int = 8
    font_scale: float = 1.2
    export_format: str = 'svg'
    show_annotations: bool = True

class DataValidator:
    """Validates data before visualization."""
    
    @staticmethod
    def validate_results_data(df: pd.DataFrame) -> bool:
        """
        Validate required columns and data types.
        
        Args:
            df: DataFrame containing optimization results
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValueError: If required columns are missing or data types are incorrect
        """
        required_columns = {
            'depeg_threshold': np.number,
            'trade_amount': np.number,
            'stop_loss': np.number,
            'take_profit': np.number,
            'sharpe_ratio': np.number,
            'total_return': np.number,
            'max_drawdown': np.number,
            'win_rate': np.number
        }
        
        for col, dtype in required_columns.items():
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            if not np.issubdtype(df[col].dtype, dtype):
                raise ValueError(f"Column {col} must be numeric")
        
        return True

class StrategyVisualizer:
    """Advanced visualization module for trading strategy optimization."""
    
    def __init__(self, results_manager, output_dir: str = 'optimization_results',
                 config: Optional[VisualizationConfig] = None):
        """
        Initialize visualization module.
        
        Args:
            results_manager: OptimizationResults instance
            output_dir: Directory for saving visualizations
            config: Visualization configuration settings
        """
        self.results = results_manager
        self.base_dir = Path(output_dir)
        self.plots_dir = self.base_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validate data
        try:
            DataValidator.validate_results_data(self.results.results)
        except ValueError as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise
            
        # Configure plotting settings
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Configure matplotlib and seaborn plotting styles."""
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette, n_colors=self.config.n_colors)
        sns.set_context("notebook", font_scale=self.config.font_scale)
        
        plt.rcParams.update({
            'figure.figsize': self.config.figsize,
            'figure.dpi': self.config.dpi,
            'savefig.bbox': 'tight',
            'savefig.format': self.config.export_format
        })
    
    def create_performance_distribution(self) -> Path:
        """
        Generate performance distribution plot.
        
        Returns:
            Path: Path to saved plot
        """
        try:
            fig, ax = plt.subplots()
            
            returns = self.results.results['total_return']
            
            # Main distribution plot
            sns.histplot(returns, kde=True, ax=ax)
            
            if self.config.show_annotations:
                # Add statistical annotations
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
                ax.text(0.95, 0.95, stats_text,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_title('Distribution des Rendements Totaux')
            ax.set_xlabel('Rendement Total (%)')
            ax.set_ylabel('Fréquence')
            
            # Save plot
            output_path = self.plots_dir / f'returns_distribution.{self.config.export_format}'
            plt.savefig(output_path)
            plt.close()
            
            self._save_plot_data(returns, 'returns_distribution')
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating performance distribution: {str(e)}")
            raise
    
    def create_robustness_violin(self) -> Path:
        """
        Generate violin plot for strategy robustness analysis.
        
        Returns:
            Path: Path to saved plot
        """
        try:
            metrics = ['win_rate', 'profit_factor', 'sharpe_ratio']
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
            
            df = pd.DataFrame(data)
            
            plt.figure(figsize=(15, 10))
            sns.violinplot(data=df, x='Parameter', y='Value', hue='Metric')
            plt.xticks(rotation=45)
            plt.title('Analyse de Robustesse des Paramètres', fontsize=14)
            
            output_path = self.plots_dir / f'robustness_violin.{self.config.export_format}'
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating robustness violin plot: {str(e)}")
            raise

    def create_3d_surface(self) -> Path:
        """
        Generate 3D surface plot for parameter optimization.
        
        Returns:
            Path: Path to saved plot
        """
        try:
            # Prepare data
            pivot = self.results.results.pivot_table(
                values='sharpe_ratio',
                index='depeg_threshold',
                columns='trade_amount',
                aggfunc='mean'
            )
            
            X, Y = np.meshgrid(pivot.columns, pivot.index)
            Z = pivot.values
            
            # Create 3D surface
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            surf = ax.plot_surface(X, Y, Z, cmap='viridis')
            
            ax.set_xlabel('Trade Amount')
            ax.set_ylabel('Depeg Threshold')
            ax.set_zlabel('Sharpe Ratio')
            
            plt.colorbar(surf)
            plt.title('Surface d\'Optimisation: Impact des Paramètres sur le Sharpe Ratio')
            
            output_path = self.plots_dir / f'parameter_surface_3d.{self.config.export_format}'
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating 3D surface plot: {str(e)}")
            raise

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
                s=self.results.results['win_rate'] * 5,  # Scale dot size by win rate
                alpha=0.6,
                cmap='viridis'
            )
            
            if self.config.show_annotations:
                # Add optimal point annotation
                best_idx = self.results.results['sharpe_ratio'].idxmax()
                best_point = self.results.results.loc[best_idx]
                
                ax.annotate('Optimal',
                           (best_point['max_drawdown'], best_point['total_return']),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(facecolor='white', alpha=0.8),
                           arrowprops=dict(arrowstyle='->'))
            
            plt.colorbar(scatter, label='Ratio de Sharpe')
            
            ax.set_xlabel('Drawdown Maximum (%)')
            ax.set_ylabel('Rendement Total (%)')
            ax.set_title('Analyse Risque-Rendement')
            
            # Add efficient frontier
            self._add_efficient_frontier(ax)
            
            output_path = self.plots_dir / f'risk_return_scatter.{self.config.export_format}'
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating risk scatter: {str(e)}")
            raise
    
    def _add_efficient_frontier(self, ax):
        """Add efficient frontier curve to risk-return plot."""
        # Sort by Sharpe ratio and get top performers
        top_performers = self.results.results.nlargest(20, 'sharpe_ratio')
        
        # Fit polynomial to top performers
        z = np.polyfit(top_performers['max_drawdown'], 
                      top_performers['total_return'], 2)
        p = np.poly1d(z)
        
        x_range = np.linspace(top_performers['max_drawdown'].min(),
                            top_performers['max_drawdown'].max(), 100)
        
        ax.plot(x_range, p(x_range), '--', color='red', 
                alpha=0.5, label='Frontier efficiente')
        ax.legend()
    
    def _save_plot_data(self, data: pd.Series, plot_name: str):
        """Save raw data used in plot for future reference."""
        data_path = self.plots_dir / f'{plot_name}_data.json'
        
        if isinstance(data, pd.Series):
            data = data.to_dict()
            
        with open(data_path, 'w') as f:
            json.dump({
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'config': {k: str(v) for k, v in self.config.__dict__.items()}
            }, f, indent=4)
    
    def create_trade_duration_analysis(self):
        """Generate trade duration analysis plots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Box plot of trade durations
        sns.boxplot(
            x='outcome',
            y='duration',
            data=self.results.results,
            ax=ax1
        )
        ax1.set_title('Distribution des Durées de Trade par Résultat')
        
        # Trade hour analysis
        hour_success = self.results.results[
            self.results.results['outcome'] == 'win']['hour'].value_counts()
        hour_failure = self.results.results[
            self.results.results['outcome'] == 'loss']['hour'].value_counts()
        
        ax2.bar(hour_success.index, hour_success.values, 
               alpha=0.5, label='Gagnants')
        ax2.bar(hour_failure.index, hour_failure.values, 
               alpha=0.5, label='Perdants')
        ax2.set_title('Distribution Horaire des Trades')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'trade_analysis.svg')
        plt.close()

    def create_optimization_dashboard(self) -> Path:
        """
        Generate HTML dashboard with interactive visualizations.
        
        Returns:
            Path: Path to saved dashboard
        """
        try:
            # Get top configurations using different metrics
            top_configs = {
                'sharpe': self.results.results.nlargest(3, 'sharpe_ratio'),
                'return_risk': self.results.results.nlargest(3, 'total_return').nsmallest(3, 'max_drawdown'),
                'stability': self.results.results.nsmallest(3, 'max_drawdown')
            }
            
            # Create HTML with embedded plots
            html_content = self._generate_dashboard_html(top_configs)
            
            output_path = self.plots_dir / 'dashboard.html'
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {str(e)}")
            raise
    
    def _create_config_html(self, configs):
        """Helper method to create HTML for configuration display."""
        html = ""
        for _, config in configs.iterrows():
            html += f"""
            <div class='config-box'>
                <h4>Configuration</h4>
                <p class='metric'>Paramètres:</p>
                <ul>
                    <li>Depeg Threshold: {config['depeg_threshold']}</li>
                    <li>Trade Amount: {config['trade_amount']}</li>
                    <li>Stop Loss: {config['stop_loss']}</li>
                    <li>Take Profit: {config['take_profit']}</li>
                </ul>
                <p class='metric'>Métriques:</p>
                <ul>
                    <li>Sharpe Ratio: {config['sharpe_ratio']:.2f}</li>
                    <li>Total Return: {config['total_return']:.2f}%</li>
                    <li>Max Drawdown: {config['max_drawdown']:.2f}%</li>
                    <li>Win Rate: {config['win_rate']:.2f}%</li>
                </ul>
            </div>
            """
        return html
    
    def generate_complete_analysis(self) -> Dict[str, Path]:
        """
        Generate all visualization components.
        
        Returns:
            Dict[str, Path]: Paths to all generated visualizations
        """
        try:
            self.logger.info("Starting complete analysis generation...")
            generated_plots = {}
            
            # Generate all plots
            generated_plots['performance'] = self.create_performance_distribution()
            self.logger.info("Performance distribution generated")
            
            generated_plots['robustness'] = self.create_robustness_violin()
            self.logger.info("Robustness violin plot generated")
            
            generated_plots['surface'] = self.create_3d_surface()
            self.logger.info("3D surface plot generated")
            
            generated_plots['risk'] = self.create_risk_scatter()
            self.logger.info("Risk-return scatter plot generated")
            
            generated_plots['trades'] = self.create_trade_duration_analysis()
            self.logger.info("Trade duration analysis generated")
            
            generated_plots['dashboard'] = self.create_optimization_dashboard()
            self.logger.info("Optimization dashboard generated")
            
            # Save configuration
            config_path = self.plots_dir / 'visualization_config.json'
            with open(config_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'config': {k: str(v) for k, v in self.config.__dict__.items()},
                    'plots': {k: str(v) for k, v in generated_plots.items()}
                }, f, indent=4)
            
            self.logger.info(f"All visualizations saved in {self.plots_dir}")
            return generated_plots
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis generation: {str(e)}")
            raise

    def _generate_dashboard_html(self, top_configs: Dict) -> str:
        """
        Generate HTML content for the dashboard.
        
        Args:
            top_configs: Dictionary containing top configurations by different metrics
            
        Returns:
            str: Complete HTML dashboard content
        """
        css = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .dashboard {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
            }
            .config-box {
                background-color: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }
            .metric {
                display: flex;
                justify-content: space-between;
                margin: 5px 0;
            }
            .metric-label {
                font-weight: bold;
                color: #555;
            }
            .metric-value {
                color: #2196F3;
            }
            h1, h2, h3 {
                color: #333;
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
            }
            .parameters {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
            }
            .timestamp {
                text-align: right;
                color: #666;
                font-size: 0.9em;
                margin-top: 20px;
            }
            .summary {
                background-color: #e3f2fd;
                padding: 15px;
                border-radius: 4px;
                margin-bottom: 20px;
            }
        </style>
        """

        # Generate summary statistics
        summary_stats = {
            'sharpe_ratio': self.results.results['sharpe_ratio'].describe(),
            'total_return': self.results.results['total_return'].describe(),
            'max_drawdown': self.results.results['max_drawdown'].describe(),
            'win_rate': self.results.results['win_rate'].describe()
        }

        html_content = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport d'Optimisation de Stratégie</title>
            {css}
        </head>
        <body>
            <div class="dashboard">
                <h1>Rapport d'Optimisation de Stratégie</h1>
                
                <div class="section summary">
                    <h2>Résumé Global</h2>
                    <div class="parameters">
                        <div class="metric">
                            <span class="metric-label">Nombre total de configurations testées:</span>
                            <span class="metric-value">{len(self.results.results)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Meilleur Sharpe Ratio:</span>
                            <span class="metric-value">{self.results.results['sharpe_ratio'].max():.2f}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Meilleur rendement:</span>
                            <span class="metric-value">{self.results.results['total_return'].max():.2f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Plus faible drawdown:</span>
                            <span class="metric-value">{self.results.results['max_drawdown'].min():.2f}%</span>
                        </div>
                    </div>
                </div>
        """

        # Add sections for each optimization criteria
        for criteria, configs in top_configs.items():
            html_content += f"""
                <div class="section">
                    <h2>Meilleures Configurations - {criteria.replace('_', ' ').title()}</h2>
            """
            
            for idx, config in configs.iterrows():
                html_content += f"""
                    <div class="config-box">
                        <h3>Configuration #{idx + 1}</h3>
                        <div class="parameters">
                            <div class="metric">
                                <span class="metric-label">Depeg Threshold:</span>
                                <span class="metric-value">{config['depeg_threshold']:.4f}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Trade Amount:</span>
                                <span class="metric-value">{config['trade_amount']:.4f}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Stop Loss:</span>
                                <span class="metric-value">{config['stop_loss']:.4f}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Take Profit:</span>
                                <span class="metric-value">{config['take_profit']:.4f}</span>
                            </div>
                        </div>
                        <h4>Performances</h4>
                        <div class="parameters">
                            <div class="metric">
                                <span class="metric-label">Sharpe Ratio:</span>
                                <span class="metric-value">{config['sharpe_ratio']:.2f}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Total Return:</span>
                                <span class="metric-value">{config['total_return']:.2f}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Max Drawdown:</span>
                                <span class="metric-value">{config['max_drawdown']:.2f}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Win Rate:</span>
                                <span class="metric-value">{config['win_rate']:.2f}%</span>
                            </div>
                        </div>
                    </div>
                """
            
            html_content += "</div>"

        # Add timestamp and close tags
        html_content += f"""
                <div class="timestamp">
                    Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """

        return html_content

@dataclass
class StrategyMetrics:
    """Data class to hold key strategy metrics."""
    total_configs: int
    sharpe_stats: Dict[str, float]
    return_stats: Dict[str, float]
    drawdown_stats: Dict[str, float]
    win_rate_stats: Dict[str, float]
    best_configs: Dict[str, pd.Series]
    parameter_correlations: Dict[str, Dict[str, float]]
    stable_zone: Dict[str, Tuple[float, float]]

class VisualizationReporter:
    """
    Generate comprehensive analysis reports for strategy optimization results.
    
    This class analyzes optimization results and generates detailed reports
    in both Markdown and HTML formats, including statistical analysis,
    parameter sensitivity, and recommendations.
    """
    
    def __init__(self, results_manager, output_dir: Path):
        """
        Initialize the reporter with results and output directory.

        Args:
            results_manager: OptimizationResults instance containing results
            output_dir: Directory where reports will be saved
        """
        self.results = results_manager
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / 'reports'
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Validate data
        self._validate_results_data()
        
        # Calculate key metrics once
        self.metrics = self._calculate_strategy_metrics()

    def _validate_results_data(self) -> None:
        """Validate that required data columns are present and valid."""
        required_columns = {
            'sharpe_ratio': float,
            'total_return': float,
            'max_drawdown': float,
            'win_rate': float,
            'depeg_threshold': float,
            'trade_amount': float,
            'stop_loss': float,
            'take_profit': float
        }
        
        for col, dtype in required_columns.items():
            if col not in self.results.results.columns:
                raise ValueError(f"Missing required column: {col}")
            if not np.issubdtype(self.results.results[col].dtype, dtype):
                raise ValueError(f"Column {col} must be of type {dtype}")

    def _calculate_strategy_metrics(self) -> StrategyMetrics:
        """Calculate all strategy metrics for reporting."""
        df = self.results.results
        
        # Calculate statistics for key metrics
        metrics = StrategyMetrics(
            total_configs=len(df),
            sharpe_stats=self._calculate_metric_stats(df, 'sharpe_ratio'),
            return_stats=self._calculate_metric_stats(df, 'total_return'),
            drawdown_stats=self._calculate_metric_stats(df, 'max_drawdown'),
            win_rate_stats=self._calculate_metric_stats(df, 'win_rate'),
            best_configs=self._find_best_configurations(df),
            parameter_correlations=self._calculate_parameter_correlations(df),
            stable_zone=self._identify_stable_zone(df)
        )
        
        self.logger.info("Strategy metrics calculated successfully")
        return metrics

    def _calculate_metric_stats(self, df: pd.DataFrame, metric: str) -> Dict[str, float]:
        """Calculate statistical measures for a given metric."""
        return {
            'min': df[metric].min(),
            'max': df[metric].max(),
            'mean': df[metric].mean(),
            'median': df[metric].median(),
            'std': df[metric].std(),
            'skew': df[metric].skew(),
            'kurtosis': df[metric].kurtosis()
        }

    def _find_best_configurations(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Find best configurations according to different criteria."""
        return {
            'sharpe': df.nlargest(1, 'sharpe_ratio').iloc[0],
            'return': df.nlargest(1, 'total_return').iloc[0],
            'risk': df.nsmallest(1, 'max_drawdown').iloc[0],
            'stability': df[
                (df['sharpe_ratio'] > df['sharpe_ratio'].mean()) &
                (df['max_drawdown'] < df['max_drawdown'].mean())
            ].nlargest(1, 'sharpe_ratio').iloc[0]
        }

    def _calculate_parameter_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between parameters and performance metrics."""
        parameters = ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        
        correlations = {}
        for param in parameters:
            correlations[param] = {
                metric: df[param].corr(df[metric])
                for metric in metrics
            }
        
        return correlations

    def _identify_stable_zone(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Identify stable parameter zones based on top performing configurations."""
        stable_configs = df[
            (df['sharpe_ratio'] > df['sharpe_ratio'].quantile(0.75)) &
            (df['max_drawdown'] < df['max_drawdown'].quantile(0.25))
        ]
        
        parameters = ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']
        return {
            param: (
                stable_configs[param].mean(),
                stable_configs[param].std()
            )
            for param in parameters
        }

    def generate_markdown_report(self) -> Path:
        """Generate comprehensive markdown report."""
        try:
            report_content = self._generate_markdown_content()
            report_path = self.report_dir / f'optimization_report_{datetime.now():%Y%m%d_%H%M%S}.md'
            
            report_path.write_text(report_content)
            self.logger.info(f"Markdown report generated: {report_path}")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating markdown report: {str(e)}")
            raise

    def _generate_markdown_content(self) -> str:
        """Generate the main markdown report content."""
        return f"""# Rapport d'Analyse d'Optimisation de Stratégie

## Résumé Exécutif

Analyse basée sur **{self.metrics.total_configs}** configurations testées.

### Métriques Principales
{self._format_key_metrics_section()}

## Configurations Optimales
{self._format_best_configs_section()}

## Analyse de Sensibilité des Paramètres
{self._format_sensitivity_section()}

## Zone de Stabilité Identifiée
{self._format_stable_zone_section()}

## Recommandations
{self._generate_recommendations()}

## Notes Techniques
{self._format_technical_notes()}

_Rapport généré le {datetime.now():%Y-%m-%d %H:%M:%S}_
"""

    def _format_key_metrics_section(self) -> str:
        """Format key metrics section of the report."""
        return f"""
            - **Sharpe Ratio**
            - Maximum: {self.metrics.sharpe_stats['max']:.2f}
            - Moyenne: {self.metrics.sharpe_stats['mean']:.2f}
            - Écart-type: {self.metrics.sharpe_stats['std']:.2f}

            - **Rendement Total**
            - Maximum: {self.metrics.return_stats['max']:.2f}%
            - Moyenne: {self.metrics.return_stats['mean']:.2f}%
            - Écart-type: {self.metrics.return_stats['std']:.2f}%

            - **Drawdown Maximum**
            - Minimum: {self.metrics.drawdown_stats['min']:.2f}%
            - Moyenne: {self.metrics.drawdown_stats['mean']:.2f}%
            - Écart-type: {self.metrics.drawdown_stats['std']:.2f}%
            """

    def _format_best_configs_section(self) -> str:
        """Format best configurations section of the report."""
        sections = []
        for config_type, config in self.metrics.best_configs.items():
            sections.append(f"""
                        ### Configuration {config_type.title()}
                        - Depeg Threshold: {config['depeg_threshold']:.4f}
                        - Trade Amount: {config['trade_amount']:.4f}
                        - Stop Loss: {config['stop_loss']:.4f}
                        - Take Profit: {config['take_profit']:.4f}

                        **Performances:**
                        - Sharpe Ratio: {config['sharpe_ratio']:.2f}
                        - Rendement Total: {config['total_return']:.2f}%
                        - Drawdown Maximum: {config['max_drawdown']:.2f}%
                        - Win Rate: {config['win_rate']:.2f}%
                        """)
        return "\n".join(sections)

    def _format_sensitivity_section(self) -> str:
        """Format parameter sensitivity section of the report."""
        sections = []
        for param, correlations in self.metrics.parameter_correlations.items():
            sections.append(f"""
                            ### {param}
                        - Impact sur Sharpe Ratio: {correlations['sharpe_ratio']:.3f}
                        - Impact sur Rendement: {correlations['total_return']:.3f}
                        - Impact sur Drawdown: {correlations['max_drawdown']:.3f}
                        - Impact sur Win Rate: {correlations['win_rate']:.3f}
                        """)
        return "\n".join(sections)

    def _format_stable_zone_section(self) -> str:
        """Format stable zone section of the report."""
        sections = []
        for param, (mean, std) in self.metrics.stable_zone.items():
            sections.append(f"- {param}: {mean:.4f} ± {std:.4f}")
        return "\n".join(sections)

    def _generate_recommendations(self) -> str:
        """Generate strategy recommendations based on analysis."""
        correlations = self.metrics.parameter_correlations
        
        # Identify most influential parameters
        param_importance = {
            param: abs(corr['sharpe_ratio'])
            for param, corr in correlations.items()
        }
        
        most_important = max(param_importance.items(), key=lambda x: x[1])[0]
        
        return f"""
                1. **Paramètre Clé**
                - Le paramètre le plus influent est `{most_important}`
                - Zone optimale: {self.metrics.stable_zone[most_important][0]:.4f} ± {self.metrics.stable_zone[most_important][1]:.4f}

                2. **Gestion du Risque**
                - Stop Loss recommandé: {self.metrics.best_configs['stability']['stop_loss']:.4f}
                - Take Profit recommandé: {self.metrics.best_configs['stability']['take_profit']:.4f}

                3. **Points d'Attention**
                - {self._generate_attention_points()}
                """

    def _generate_attention_points(self) -> str:
        """Generate key points requiring attention."""
        points = []
        metrics = self.metrics
        
        if metrics.drawdown_stats['max'] > 20:
            points.append("Risque de drawdown important > 20%")
        
        if metrics.sharpe_stats['std'] > 1:
            points.append("Forte sensibilité aux paramètres (Sharpe ratio instable)")
        
        if metrics.win_rate_stats['mean'] < 50:
            points.append("Win rate moyen faible, optimisation de la précision nécessaire")
        
        if not points:
            points.append("Aucun point critique identifié")
            
        return "\n   - ".join(points)

    def _format_technical_notes(self) -> str:
        """Format technical notes section."""
        df = self.results.results
        return f"""
        - Période d'analyse: {df.index.min()} à {df.index.max()}
        - Configurations testées: {len(df)}
        - Distribution des trades:
        - Win Rate moyen: {self.metrics.win_rate_stats['mean']:.2f}%
        - Écart-type Win Rate: {self.metrics.win_rate_stats['std']:.2f}%
        """