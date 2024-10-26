import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader, select_autoescape

@dataclass
class ReportMetrics:
    """Data class to hold report metrics with proper typing."""
    total_configs: int
    sharpe_stats: Dict[str, float]
    return_stats: Dict[str, float]
    drawdown_stats: Dict[str, float]
    win_rate_stats: Dict[str, float]
    best_configs: Dict[str, Dict[str, float]]
    parameter_correlations: Dict[str, Dict[str, float]]
    timestamp: datetime = datetime.now()

class ReportGenerator:
    """Report generator with proper encoding and formatting."""
    
    def __init__(self, results_manager, output_dir: Path):
        """
        Initialize the report generator.
        
        Args:
            results_manager: Results manager containing optimization data
            output_dir: Directory for saving reports
        """
        self.results = results_manager
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / 'reports'
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Jinja2 environment
        template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
            encoding='utf-8'
        )
        
        # Load and validate data
        self.metrics = self._calculate_metrics()
        
    def _calculate_metrics(self) -> ReportMetrics:
        """Calculate all metrics needed for the report."""
        try:
            df = self.results.results
            
            return ReportMetrics(
                total_configs=len(df),
                sharpe_stats=self._calculate_metric_stats(df, 'sharpe_ratio'),
                return_stats=self._calculate_metric_stats(df, 'total_return'),
                drawdown_stats=self._calculate_metric_stats(df, 'max_drawdown'),
                win_rate_stats=self._calculate_metric_stats(df, 'win_rate'),
                best_configs=self._find_best_configurations(df),
                parameter_correlations=self._calculate_correlations(df)
            )
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def _calculate_metric_stats(self, df: pd.DataFrame, metric: str) -> Dict[str, float]:
        """Calculate statistical measures for a given metric."""
        try:
            series = df[metric].copy()
            return {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'median': float(series.median()),
                'p25': float(series.quantile(0.25)),
                'p75': float(series.quantile(0.75))
            }
        except Exception as e:
            self.logger.error(f"Error calculating stats for {metric}: {str(e)}")
            return {
                'min': 0.0, 'max': 0.0, 'mean': 0.0,
                'std': 0.0, 'median': 0.0, 'p25': 0.0, 'p75': 0.0
            }

    def _find_best_configurations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Find best configurations according to different criteria."""
        try:
            best_configs = {}
            metrics = {
                'sharpe': ('sharpe_ratio', 'max'),
                'return': ('total_return', 'max'),
                'risk': ('max_drawdown', 'min')
            }
            
            for config_name, (metric, criterion) in metrics.items():
                idx = df[metric].idxmax() if criterion == 'max' else df[metric].idxmin()
                config = df.loc[idx]
                
                best_configs[config_name] = {
                    'depeg_threshold': float(config['depeg_threshold']),
                    'trade_amount': float(config['trade_amount']),
                    'stop_loss': float(config['stop_loss']),
                    'take_profit': float(config['take_profit']),
                    'sharpe_ratio': float(config['sharpe_ratio']),
                    'total_return': float(config['total_return']),
                    'max_drawdown': float(config['max_drawdown']),
                    'win_rate': float(config['win_rate'])
                }
                
            return best_configs
            
        except Exception as e:
            self.logger.error(f"Error finding best configurations: {str(e)}")
            return {}

    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate parameter correlations."""
        try:
            parameters = ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']
            metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
            
            correlations = {}
            for param in parameters:
                correlations[param] = {}
                for metric in metrics:
                    correlations[param][metric] = float(df[param].corr(df[metric]))
                    
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {str(e)}")
            return {}

    def generate_markdown_report(self) -> Path:
        """Generate a markdown report with proper encoding."""
        try:
            template = self.jinja_env.get_template('strategy_report.md')
            
            report_content = template.render(
                metrics=self.metrics,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                format_number=lambda x: f"{x:.2f}",
                format_percentage=lambda x: f"{x:.2f}%"
            )
            
            report_path = self.report_dir / f'optimization_report_{datetime.now():%Y%m%d_%H%M%S}.md'
            report_path.write_text(report_content, encoding='utf-8')
            
            self.logger.info(f"Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            raise

    def generate_html_report(self) -> Path:
        """Generate an HTML report with proper encoding."""
        try:
            template = self.jinja_env.get_template('strategy_report.html')
            
            report_content = template.render(
                metrics=self.metrics,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                format_number=lambda x: f"{x:.2f}",
                format_percentage=lambda x: f"{x:.2f}%"
            )
            
            report_path = self.report_dir / f'optimization_report_{datetime.now():%Y%m%d_%H%M%S}.html'
            report_path.write_text(report_content, encoding='utf-8')
            
            self.logger.info(f"HTML report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {str(e)}")
            raise

    def export_metrics_json(self) -> Path:
        """Export metrics to JSON with proper encoding."""
        try:
            metrics_dict = {
                'total_configs': self.metrics.total_configs,
                'sharpe_stats': self.metrics.sharpe_stats,
                'return_stats': self.metrics.return_stats,
                'drawdown_stats': self.metrics.drawdown_stats,
                'win_rate_stats': self.metrics.win_rate_stats,
                'best_configs': self.metrics.best_configs,
                'parameter_correlations': self.metrics.parameter_correlations,
                'timestamp': self.metrics.timestamp.isoformat()
            }
            
            json_path = self.report_dir / f'metrics_{datetime.now():%Y%m%d_%H%M%S}.json'
            json_path.write_text(json.dumps(metrics_dict, indent=2, ensure_ascii=False), encoding='utf-8')
            
            return json_path
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics to JSON: {str(e)}")
            raise