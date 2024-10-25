"""Metrics calculation and storage utilities for strategy optimization."""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    """Data class to hold key strategy metrics.
    
    Attributes:
        total_configs (int): Total number of configurations tested
        sharpe_stats (Dict[str, float]): Statistics for Sharpe ratio
        return_stats (Dict[str, float]): Statistics for total returns
        drawdown_stats (Dict[str, float]): Statistics for maximum drawdown
        win_rate_stats (Dict[str, float]): Statistics for win rate
        best_configs (Dict[str, pd.Series]): Best configurations by different criteria
        parameter_correlations (Dict[str, Dict[str, float]]): Parameter correlation matrices
        stable_zone (Dict[str, Tuple[float, float]]): Stable parameter zones
    """
    total_configs: int
    sharpe_stats: Dict[str, float]
    return_stats: Dict[str, float]
    drawdown_stats: Dict[str, float]
    win_rate_stats: Dict[str, float]
    best_configs: Dict[str, pd.Series]
    parameter_correlations: Dict[str, Dict[str, float]]
    stable_zone: Dict[str, Tuple[float, float]]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'total_configs': self.total_configs,
            'sharpe_stats': self.sharpe_stats,
            'return_stats': self.return_stats,
            'drawdown_stats': self.drawdown_stats,
            'win_rate_stats': self.win_rate_stats,
            'best_configs': {k: v.to_dict() for k, v in self.best_configs.items()},
            'parameter_correlations': self.parameter_correlations,
            'stable_zone': {k: list(v) for k, v in self.stable_zone.items()},
            'timestamp': self.timestamp.isoformat()
        }

class MetricsCalculator:
    """Handles calculation of strategy metrics."""
    
    # Class constants
    PARAMETERS = ['depeg_threshold', 'trade_amount', 'stop_loss', 'take_profit']
    METRICS = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
    
    @staticmethod
    def calculate_metric_stats(df: pd.DataFrame, metric: str) -> Dict[str, float]:
        """Calculate statistical measures for a given metric.
        
        Args:
            df (pd.DataFrame): DataFrame containing the metric
            metric (str): Name of the metric to analyze
            
        Returns:
            Dict[str, float]: Dictionary of statistical measures
            
        Raises:
            KeyError: If metric is not found in DataFrame
            ValueError: If metric contains invalid values
        """
        try:
            if metric not in df.columns:
                raise KeyError(f"Metric '{metric}' not found in DataFrame")
                
            series = df[metric].copy()
            
            # Handle NaN values
            if series.isna().any():
                logger.warning(f"NaN values found in {metric}. Removing for calculations.")
                series = series.dropna()
            
            if len(series) == 0:
                raise ValueError(f"No valid values found for metric '{metric}'")

            stats = {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'skew': float(series.skew()),
                'kurtosis': float(series.kurtosis())
            }
            
            # Add percentiles
            for p in [25, 50, 75]:
                stats[f'percentile_{p}'] = float(series.quantile(p/100))
            
            return stats

        except Exception as e:
            logger.error(f"Error calculating stats for {metric}: {str(e)}")
            raise
    
    @classmethod
    def calculate_parameter_correlations(cls, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between parameters and performance metrics.
        
        Args:
            df (pd.DataFrame): DataFrame containing parameters and metrics
            
        Returns:
            Dict[str, Dict[str, float]]: Correlation matrices
            
        Raises:
            ValueError: If required columns are missing
        """
        try:
            # Validate columns
            missing_params = set(cls.PARAMETERS) - set(df.columns)
            missing_metrics = set(cls.METRICS) - set(df.columns)
            
            if missing_params or missing_metrics:
                raise ValueError(
                    f"Missing columns. Parameters: {missing_params}, Metrics: {missing_metrics}"
                )
            
            correlations = {}
            for param in cls.PARAMETERS:
                correlations[param] = {}
                for metric in cls.METRICS:
                    try:
                        # Handle zero variance case
                        if df[param].std() == 0 or df[metric].std() == 0:
                            corr = np.nan
                        else:
                            corr = df[param].corr(df[metric])
                        correlations[param][metric] = float(corr)
                    except Exception as e:
                        logger.warning(f"Could not calculate correlation between {param} and {metric}: {e}")
                        correlations[param][metric] = np.nan
            
            return correlations

        except Exception as e:
            logger.error(f"Error calculating parameter correlations: {str(e)}")
            raise

    @classmethod
    def calculate_all_metrics(cls, df: pd.DataFrame) -> StrategyMetrics:
        """Calculate all metrics for the strategy.
        
        Args:
            df (pd.DataFrame): DataFrame containing optimization results
            
        Returns:
            StrategyMetrics: Complete set of calculated metrics
        """
        try:
            metrics = StrategyMetrics(
                total_configs=len(df),
                sharpe_stats=cls.calculate_metric_stats(df, 'sharpe_ratio'),
                return_stats=cls.calculate_metric_stats(df, 'total_return'),
                drawdown_stats=cls.calculate_metric_stats(df, 'max_drawdown'),
                win_rate_stats=cls.calculate_metric_stats(df, 'win_rate'),
                best_configs={},  # To be filled by optimization module
                parameter_correlations=cls.calculate_parameter_correlations(df),
                stable_zone={},  # To be filled by optimization module
            )
            
            logger.info("Successfully calculated all metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise