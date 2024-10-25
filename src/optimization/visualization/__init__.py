"""Visualization package initialization.

This module provides a collection of tools for visualizing and analyzing optimization results.
"""

# Base configuration and validators
from .config import VisualizationConfig
from .data_validator import DataValidator

# Core metrics and calculations
from .metrics import StrategyMetrics, MetricsCalculator

# Visualization components
from .base_visualizer import BaseVisualizer
from .strategy_visualizer import StrategyVisualizer

# Report generation
from .report_generator import ReportGenerator

__all__ = [
    # Configuration and validation
    'VisualizationConfig',
    'DataValidator',
    
    # Metrics
    'StrategyMetrics',
    'MetricsCalculator',
    
    # Visualization
    'BaseVisualizer',
    'StrategyVisualizer',
    
    # Reporting
    'ReportGenerator'
]

__version__ = '1.0.0'