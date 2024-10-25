import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from .metrics import StrategyMetrics, MetricsCalculator

class ReportGenerator:
    """Generate comprehensive analysis reports."""
    
    def __init__(self, results_manager, output_dir: Path):
        self.results = results_manager
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / 'reports'
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Debug log the available columns
        self.logger.info(f"Available columns: {list(self.results.results.columns)}")
        
        self.metrics = self._calculate_strategy_metrics()

    def _find_best_configurations(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Find best configurations according to different criteria."""
        try:
            best_configs = {}
            
            if df.empty:
                self.logger.warning("Empty DataFrame provided")
                return best_configs

            required_cols = ['sharpe_ratio', 'total_return', 'max_drawdown']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
                return best_configs

            # Find best configuration for each metric
            metrics_to_check = [
                ('sharpe', 'sharpe_ratio', 'nlargest'),
                ('return', 'total_return', 'nlargest'),
                ('risk', 'max_drawdown', 'nsmallest')
            ]

            for config_name, metric, func in metrics_to_check:
                try:
                    metric_df = getattr(df, func)(1, metric)
                    if not metric_df.empty:
                        best_configs[config_name] = metric_df.iloc[0]
                except Exception as e:
                    self.logger.warning(f"Failed to find best {config_name}: {str(e)}")

            # Find stability configuration
            try:
                stable_mask = (
                    (df['sharpe_ratio'] > df['sharpe_ratio'].mean()) &
                    (df['max_drawdown'] < df['max_drawdown'].mean())
                )
                stable_configs = df[stable_mask]
                if not stable_configs.empty:
                    best_configs['stability'] = stable_configs.nlargest(1, 'sharpe_ratio').iloc[0]
            except Exception as e:
                self.logger.warning(f"Failed to find stability configuration: {str(e)}")

            return best_configs

        except Exception as e:
            self.logger.error(f"Error in best configurations: {str(e)}")
            return {}        
    
    def _identify_stable_zone(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        Identify stable parameter zones based on top performing configurations.

        Args:
            df (pd.DataFrame): DataFrame containing optimization results

        Returns:
            Dict[str, Tuple[float, float]]: Parameter zones with (mean, std) for each parameter
        """
        try:
            # Filter for top performing configurations
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

        except Exception as e:
            self.logger.error(f"Error identifying stable zones: {str(e)}")
            return {}

    def _calculate_strategy_metrics(self) -> StrategyMetrics:
        """Calculate all strategy metrics for reporting."""
        try:
            df = self.results.results
            calc = MetricsCalculator()
            
            metrics = StrategyMetrics(
                total_configs=len(df),
                sharpe_stats=calc.calculate_metric_stats(df, 'sharpe_ratio'),
                return_stats=calc.calculate_metric_stats(df, 'total_return'),
                drawdown_stats=calc.calculate_metric_stats(df, 'max_drawdown'),
                win_rate_stats=calc.calculate_metric_stats(df, 'win_rate'),
                best_configs=self._find_best_configurations(df),
                parameter_correlations=calc.calculate_parameter_correlations(df),
                stable_zone=self._identify_stable_zone(df)
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating strategy metrics: {str(e)}")
            raise

    def _format_key_metrics_section(self) -> str:
        """Format key metrics section of the report."""
        return f"""
### Métriques de Performance

#### Ratio de Sharpe
- Maximum: {self.metrics.sharpe_stats['max']:.2f}
- Moyenne: {self.metrics.sharpe_stats['mean']:.2f}
- Écart-type: {self.metrics.sharpe_stats['std']:.2f}

#### Rendement Total
- Maximum: {self.metrics.return_stats['max']:.2f}%
- Moyenne: {self.metrics.return_stats['mean']:.2f}%
- Écart-type: {self.metrics.return_stats['std']:.2f}%

#### Drawdown Maximum
- Minimum: {self.metrics.drawdown_stats['min']:.2f}%
- Moyenne: {self.metrics.drawdown_stats['mean']:.2f}%
- Écart-type: {self.metrics.drawdown_stats['std']:.2f}%
"""

    def _format_best_configs_section(self) -> str:
        """Format best configurations section of the report."""
        try:
            sections = []
            
            if not self.metrics.best_configs:
                return "Aucune configuration optimale n'a pu être déterminée."

            for config_type, config in self.metrics.best_configs.items():
                try:
                    section = f"\n### Configuration {config_type.title()}\n"
                    
                    # Add parameters if they exist
                    for param in ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']:
                        if param in config:
                            section += f"- {param}: {config[param]:.4f}\n"
                    
                    # Add performance metrics if they exist
                    section += "\n**Performances:**\n"
                    metrics_map = {
                        'sharpe_ratio': 'Sharpe Ratio',
                        'total_return': 'Rendement Total',
                        'max_drawdown': 'Drawdown Maximum',
                        'win_rate': 'Win Rate'
                    }
                    
                    for metric_key, metric_name in metrics_map.items():
                        if metric_key in config:
                            value = config[metric_key]
                            suffix = '%' if metric_key in ['total_return', 'max_drawdown', 'win_rate'] else ''
                            section += f"- {metric_name}: {value:.2f}{suffix}\n"
                    
                    sections.append(section)
                except Exception as e:
                    self.logger.warning(f"Error formatting config {config_type}: {e}")
                    continue

            return "\n".join(sections) if sections else "Aucune configuration à afficher."

        except Exception as e:
            self.logger.error(f"Error formatting best configurations: {str(e)}")
            return "Erreur lors de la génération des configurations optimales."

    def generate_markdown_report(self) -> Path:
        """Generate comprehensive markdown report."""
        try:
            sections = [
                self._generate_header(),
                self._generate_executive_summary(),
                self._format_key_metrics_section(),
                self._format_best_configs_section(),
                self._format_stable_zone_section(),
                self._generate_recommendations(),
                self._generate_footer()
            ]
            
            report_content = "\n\n".join(sec for sec in sections if sec)
            report_path = self.report_dir / f'optimization_report_{datetime.now():%Y%m%d_%H%M%S}.md'
            
            report_path.write_text(report_content)
            self.logger.info(f"Report generated successfully: {report_path}")
            
            return report_path

        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            raise

    def _format_stable_zone_section(self) -> str:
        """Format stable zone section of the report."""
        sections = []
        for param, (mean, std) in self.metrics.stable_zone.items():
            sections.append(f"- {param}: {mean:.4f} ± {std:.4f}")
        return "\n".join(sections)

    def _generate_recommendations(self) -> str:
        """Generate strategy recommendations based on analysis."""
        try:
            recommendations = ["## Recommandations\n"]
            
            # 1. Parameter importance
            if self.metrics.parameter_correlations:
                try:
                    param_importance = {
                        param: abs(corr.get('sharpe_ratio', 0))
                        for param, corr in self.metrics.parameter_correlations.items()
                    }
                    if param_importance:
                        most_important = max(param_importance.items(), key=lambda x: x[1])[0]
                        recommendations.append(f"1. **Paramètre Clé**\n   - Le paramètre le plus influent est `{most_important}`")
                        
                        if most_important in self.metrics.stable_zone:
                            mean, std = self.metrics.stable_zone[most_important]
                            recommendations.append(f"   - Zone optimale: {mean:.4f} ± {std:.4f}\n")
                except Exception as e:
                    self.logger.warning(f"Could not determine parameter importance: {e}")

            # 2. Risk Management
            try:
                stability_config = self.metrics.best_configs.get('stability', {})
                if 'stop_loss' in stability_config and 'take_profit' in stability_config:
                    recommendations.append("2. **Gestion du Risque**")
                    recommendations.append(f"   - Stop Loss recommandé: {stability_config['stop_loss']:.4f}")
                    recommendations.append(f"   - Take Profit recommandé: {stability_config['take_profit']:.4f}\n")
            except Exception as e:
                self.logger.warning(f"Could not generate risk management recommendations: {e}")

            # 3. Performance Metrics
            try:
                recommendations.append("3. **Observations**")
                metrics_available = False
                
                if hasattr(self.metrics, 'sharpe_stats') and self.metrics.sharpe_stats:
                    recommendations.append(f"   - Ratio de Sharpe moyen: {self.metrics.sharpe_stats['mean']:.2f}")
                    metrics_available = True
                
                if hasattr(self.metrics, 'drawdown_stats') and self.metrics.drawdown_stats:
                    recommendations.append(f"   - Drawdown maximum moyen: {self.metrics.drawdown_stats['mean']:.2f}%")
                    metrics_available = True
                
                if hasattr(self.metrics, 'win_rate_stats') and self.metrics.win_rate_stats:
                    recommendations.append(f"   - Win Rate moyen: {self.metrics.win_rate_stats['mean']:.2f}%")
                    metrics_available = True
                
                if not metrics_available:
                    recommendations.append("   - Données insuffisantes pour générer des observations")
            except Exception as e:
                self.logger.warning(f"Could not generate performance observations: {e}")

            return "\n".join(recommendations)

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return "Impossible de générer des recommandations en raison d'une erreur."
        
    def _generate_header(self) -> str:
        """Generate report header."""
        return "# Rapport d'Analyse d'Optimisation de Stratégie"

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        return f"""## Résumé Exécutif
Analyse basée sur **{self.metrics.total_configs}** configurations testées."""

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"_Rapport généré le {datetime.now():%Y-%m-%d %H:%M:%S}_"

    def _calculate_metrics_safely(self, func, *args, fallback=None):
        """Safely execute a metrics calculation function."""
        try:
            return func(*args)
        except Exception as e:
            self.logger.warning(f"Failed to calculate metric: {str(e)}")
            return fallback

    def _format_metric_value(self, value: float, metric: str) -> str:
        """Format a metric value with appropriate suffix."""
        if value is None:
            return "N/A"
        
        metric_info = self.METRICS.get(metric, {'suffix': ''})
        suffix = metric_info['suffix']
        return f"{value:.2f}{suffix}"
