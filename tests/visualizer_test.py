import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.visualizer import OptimizationVisualizer

# ------------------------- Fixtures -------------------------

@pytest.fixture
def mock_results_manager():
    """Fixture for a mock results manager."""
    mock = Mock()
    # Mocking get_parameter_performance method
    mock.get_parameter_performance.return_value = {
        'value_stats': {
            'param1': {'mean': 10, 'std': 2},
            'param2': {'mean': 20, 'std': 4},
        },
        'best_value': 'param1',
        'worst_value': 'param2'
    }
    # Mocking calculate_correlations method
    mock.calculate_correlations.return_value = pd.DataFrame(
        np.random.rand(5, 5),
        columns=[f'param_{i}' for i in range(5)],
        index=[f'param_{i}' for i in range(5)]
    )
    # Mocking analyze_parameter_sensitivity method
    mock.analyze_parameter_sensitivity.return_value = {
        'sensitivity_by_range': [
            {'param_range': (0, 10), 'sensitivity': 0.5},
            {'param_range': (10, 20), 'sensitivity': 0.3},
        ]
    }
    # Mocking results and metric_columns
    mock.results = pd.DataFrame({'metric1': np.random.rand(100)})
    mock.metric_columns = ['metric1']
    mock.param_columns = ['param1', 'param2']
    return mock

@pytest.fixture
def visualizer(mock_results_manager, tmp_path):
    """Fixture for the OptimizationVisualizer instance."""
    return OptimizationVisualizer(results_manager=mock_results_manager, output_dir=tmp_path)

# ------------------------- Test Cases -------------------------

def test_initialization(visualizer, tmp_path):
    """Test that the visualizer initializes correctly."""
    assert visualizer.output_dir == Path(tmp_path)
    assert visualizer.default_figsize == (10, 6)
    assert visualizer.default_dpi == 100

def test_plot_parameter_performance(visualizer, mock_results_manager):
    """Test parameter performance plot creation."""
    output_path = visualizer.plot_parameter_performance(param_name='param1', metric_name='metric1')
    assert os.path.exists(output_path)
    mock_results_manager.get_parameter_performance.assert_called_once_with('param1', 'metric1')

def test_plot_correlation_heatmap(visualizer, mock_results_manager):
    """Test correlation heatmap creation."""
    output_path = visualizer.plot_correlation_heatmap()
    assert os.path.exists(output_path)
    mock_results_manager.calculate_correlations.assert_called_once()

def test_plot_metric_distributions(visualizer, mock_results_manager):
    """Test metric distribution plot creation."""
    plot_paths = visualizer.plot_metric_distributions(metrics=['metric1'])
    assert 'metric1' in plot_paths
    assert os.path.exists(plot_paths['metric1'])
    assert mock_results_manager.results['metric1'].mean() is not None

def test_plot_parameter_sensitivity(visualizer, mock_results_manager):
    """Test parameter sensitivity plot creation."""
    output_path = visualizer.plot_parameter_sensitivity(param_name='param1', metric_name='metric1')
    assert os.path.exists(output_path)
    mock_results_manager.analyze_parameter_sensitivity.assert_called_once_with('param1', 'metric1')

def test_create_optimization_report(visualizer, mock_results_manager):
    """Test optimization report creation."""
    output_dir = visualizer.create_optimization_report()
    assert os.path.exists(output_dir)
    mock_results_manager.calculate_correlations.assert_called_once()
    assert mock_results_manager.get_parameter_performance.call_count == len(mock_results_manager.param_columns) * len(mock_results_manager.metric_columns)
    assert mock_results_manager.analyze_parameter_sensitivity.call_count == len(mock_results_manager.param_columns) * len(mock_results_manager.metric_columns)

def test_set_style(visualizer):
    """Test setting a new style for plots."""
    visualizer.set_style(style='ggplot', figsize=(8, 5), dpi=120)
    assert visualizer.default_figsize == (8, 5)
    assert visualizer.default_dpi == 120
