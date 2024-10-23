import os
import sys
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from contextlib import ExitStack
import time

# Add src path to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.optimizer import StrategyOptimizer

# ------------------------------------ Fixtures ------------------------------------

@pytest.fixture
def valid_param_ranges():
    """Fixture for valid parameter ranges."""
    return {
        'depeg_threshold': {'start': 2.0, 'end': 6.0, 'step': 0.5},
        'trade_amount': {'start': 0.05, 'end': 0.2, 'step': 0.05}
    }

@pytest.fixture
def valid_base_config(tmp_path):
    """Fixture for valid base configuration with a temporary data file."""
    data_file = tmp_path / "test_data.csv"
    data_file.write_text("timestamp,close\n1,100\n2,101\n")
    
    return {
        'asset': 'EUTEUR',
        'data_path': str(data_file),
        'initial_cash': 10000,
        'base_currency': 'EUR'
    }

@pytest.fixture
def sample_test_data():
    """Fixture for sample optimization results."""
    return {
        'parameters': {
            'depeg_threshold': 2.5,
            'trade_amount': 0.1
        },
        'metrics': {
            'total_return': 15.5,
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.12,
            'win_rate': 65.0
        }
    }

@pytest.fixture
def strategy_optimizer(valid_param_ranges, valid_base_config, tmp_path):
    """Fixture for StrategyOptimizer instance with temporary output directory."""
    output_dir = tmp_path / "optimization_output"
    optimizer = StrategyOptimizer(
        param_ranges=valid_param_ranges,
        base_config=valid_base_config,
        output_dir=str(output_dir)
    )
    # Create plots directory
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)
    return optimizer

@pytest.fixture
def optimizer_with_data(strategy_optimizer, sample_test_data):
    """Fixture for StrategyOptimizer instance with pre-loaded test data."""
    strategy_optimizer.results.add_result(
        params=sample_test_data['parameters'],
        metrics=sample_test_data['metrics']
    )
    return strategy_optimizer

# -------------------------------- Initialization Tests ---------------------------

class TestInitialization:
    """Tests for StrategyOptimizer initialization."""
    
    def test_valid_initialization(self, strategy_optimizer, valid_param_ranges, valid_base_config):
        """Test initialization with valid parameters."""
        assert strategy_optimizer.param_grid is not None
        assert strategy_optimizer.results is not None
        assert strategy_optimizer.visualizer is not None
        assert strategy_optimizer.base_config == valid_base_config
        assert strategy_optimizer.output_dir.exists()

    def test_output_directory_creation(self, strategy_optimizer):
        """Test output directory is created properly."""
        assert strategy_optimizer.output_dir.exists()
        plots_dir = strategy_optimizer.output_dir / 'plots'
        assert plots_dir.exists(), f"Plots directory not found at {plots_dir}"

    def test_initialization_invalid_config(self, valid_param_ranges):
        """Test initialization with invalid configurations."""
        invalid_configs = [
            {'asset': 'EUTEUR', 'initial_cash': 10000},  # Missing data_path
            {'asset': 'EUTEUR', 'data_path': 'test.csv', 'initial_cash': -1000, 'base_currency': 'EUR'},  # Negative cash
            {'asset': 'EUTEUR', 'data_path': 'nonexistent.csv', 'initial_cash': 10000, 'base_currency': 'EUR'}  # Bad path
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                StrategyOptimizer(param_ranges=valid_param_ranges, base_config=config)

# -------------------------------- Optimization Tests ---------------------------

class TestOptimizationExecution:
    """Tests for the optimization execution process."""

    def test_empty_optimization_results(self, strategy_optimizer):
        """Test handling of empty optimization results."""
        # Create empty DataFrame for results
        strategy_optimizer.results.results = pd.DataFrame()
        strategy_optimizer.results.param_columns = []
        strategy_optimizer.results.metric_columns = []

        # Patch all methods that might be called with empty results
        patches = [
            patch('src.optimization.optimizer.create_optimization_tasks'),
            patch('src.optimization.optimizer.monitor_task_progress'),
            patch('src.optimization.visualizer.OptimizationVisualizer.create_optimization_report'),
            patch.object(strategy_optimizer.results, 'calculate_statistics', return_value={}),
            patch.object(strategy_optimizer.results, 'get_summary', return_value={}),
            patch.object(strategy_optimizer.results, 'calculate_correlations', return_value=pd.DataFrame()),
            patch.object(strategy_optimizer.results, 'get_top_results', return_value=pd.DataFrame()),
            patch.object(strategy_optimizer.param_grid, 'get_param_info', return_value={}),
            patch.object(strategy_optimizer, '_create_optimization_summary', return_value={
                'runtime': {'total_seconds': 0, 'average_per_combination': 0},
                'parameters': {},
                'file_locations': {'base_directory': str(strategy_optimizer.output_dir)}
            })
        ]
        
        with ExitStack() as stack:
            # Enter all patches
            mocks = [stack.enter_context(p) for p in patches]
            mock_create = mocks[0]
            
            # Setup mock task group
            mock_task = Mock()
            mock_task.tasks = []
            mock_task.get.return_value = []
            
            mock_optimization = Mock()
            mock_optimization.apply_async.return_value = mock_task
            mock_create.return_value = mock_optimization
            
            # Run optimization
            summary = strategy_optimizer.run_optimization()
            
            # Verify empty results handling
            assert len(strategy_optimizer.results.results) == 0
            assert isinstance(summary, dict)
            assert 'runtime' in summary
            assert summary.get('parameters', {}) == {}
            assert mock_create.called
            assert not mocks[2].called  # Visualization should not be called for empty results

    def test_empty_results_summary(self, strategy_optimizer):
        """Test summary creation with empty results."""
        # Set up empty results
        strategy_optimizer.results.results = pd.DataFrame()
        strategy_optimizer.results.param_columns = []
        strategy_optimizer.results.metric_columns = []
        
        # Mock methods that might cause division by zero
        with patch.object(strategy_optimizer.results, 'calculate_statistics', return_value={}), \
             patch.object(strategy_optimizer.results, 'get_summary', return_value={}), \
             patch.object(strategy_optimizer.param_grid, 'get_param_info', return_value={}):
            
            # Create a minimal summary with empty results
            summary = strategy_optimizer._create_optimization_summary(
                start_time=time.time(),
                optimization_results=[]
            )
            
            assert isinstance(summary, dict)
            assert 'runtime' in summary
            assert 'parameters' in summary
            assert 'file_locations' in summary
            
            # Verify runtime calculation doesn't cause division by zero
            assert isinstance(summary['runtime'], dict)
            assert 'total_seconds' in summary['runtime']
            assert 'average_per_combination' in summary['runtime']
            assert summary['runtime']['average_per_combination'] == 0  # Should be 0 for empty results

# -------------------------------- Results Management Tests --------------------

class TestResultsManagement:
    """Tests for results management functionality."""

    def test_create_optimization_summary(self, optimizer_with_data):
        """Test creation of optimization summary."""
        start_time = time.time() - 100
        optimization_results = [{
            'parameters': {
                'depeg_threshold': 2.5,
                'trade_amount': 0.1
            },
            'metrics': {
                'total_return': 15.5,
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.12,
                'win_rate': 65.0
            }
        }]
        
        summary = optimizer_with_data._create_optimization_summary(
            start_time=start_time,
            optimization_results=optimization_results
        )
        
        assert 'runtime' in summary
        assert 'parameters' in summary
        assert 'file_locations' in summary
