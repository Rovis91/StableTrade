import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
import tempfile

# Correct imports with proper paths
from src.optimization.optimizer import StrategyOptimizer, OptimizationResults
from src.optimization.task_coordinator import TaskCoordinator, OptimizationConfig
from src.optimization.visualization.strategy_visualizer import StrategyVisualizer
from src.strategy.base_strategy import Strategy


class MockStrategy(Strategy):
    """Mock strategy for testing."""
    def __init__(self, market: str, trade_manager, **kwargs):
        super().__init__(market, trade_manager)
        self.config = {
            'market_type': 'spot',
            'fees': {'entry': 0.001, 'exit': 0.001},
            'max_trades': 5,
            'max_exposure': 0.5
        }
        for key, value in kwargs.items():
            setattr(self, key, value)

    def generate_signal(self, market_data, active_trades):
        return {'action': 'buy', 'amount': 0.1}

    def get_required_indicators(self):
        return {'SMA': [20]}

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1H')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(0.99, 1.01, len(dates)),
        'high': np.random.uniform(0.99, 1.01, len(dates)),
        'low': np.random.uniform(0.99, 1.01, len(dates)),
        'close': np.random.uniform(0.99, 1.01, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    })

@pytest.fixture
def optimizer_config():
    """Create sample optimizer configuration."""
    return {
        'param_ranges': {
            'threshold': {'start': 1.0, 'end': 2.0, 'step': 0.1},
            'amount': {'start': 0.1, 'end': 0.3, 'step': 0.1}
        },
        'base_config': {
            'data_path': 'test_data.csv',
            'asset': 'EUTEUR',
            'base_currency': 'EUR',
            'initial_cash': 1000
        }
    }


def test_strategy_optimizer_initialization(optimizer_config, temp_dir):
    """Test StrategyOptimizer initialization."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = StrategyOptimizer(
        strategy_class=MockStrategy,
        param_ranges=optimizer_config['param_ranges'],
        base_config=optimizer_config['base_config'],
        output_dir=str(output_dir)
    )
    
    assert optimizer.strategy_class == MockStrategy
    assert optimizer.param_ranges == optimizer_config['param_ranges']
    assert optimizer.base_config == optimizer_config['base_config']
    assert Path(optimizer.output_dir).exists()


def test_task_coordinator_task_submission(optimizer_config, temp_dir):
    """Test TaskCoordinator task submission."""
    config = OptimizationConfig(
        strategy_class=MockStrategy,
        param_ranges=optimizer_config['param_ranges'],
        data_path=optimizer_config['base_config']['data_path'],
        base_config=optimizer_config['base_config'],
        output_dir=temp_dir,
        max_parallel_tasks=2
    )
    
    coordinator = TaskCoordinator(config)
    
    with patch('src.optimization.celery_tasks.run_optimization') as mock_run:
        mock_result = {
            'parameters': {'threshold': 1.0},
            'metrics': {'sharpe_ratio': 1.5},
            'execution_time': 1.0
        }
        mock_run.return_value = mock_result
        
        result = coordinator._run_single_optimization({'threshold': 1.0})
        assert result['metrics']['sharpe_ratio'] == 1.5

def test_strategy_visualizer(sample_data, temp_dir):
    """Test StrategyVisualizer plot generation."""
    visualizer = StrategyVisualizer(output_dir=temp_dir)
    
    # Create correct data structure
    results_df = pd.DataFrame([{
        'depeg_threshold': 1.0,
        'trade_amount': 0.1,
        'stop_loss': 0.02,
        'take_profit': 0.05,
        'sharpe_ratio': 1.5,
        'total_return': 0.05,
        'max_drawdown': 0.02,
        'win_rate': 0.6
    }])
    
    plots = visualizer.generate_all_plots(results_df)
    assert len(plots) > 0
    assert all(Path(p).exists() for p in plots.values())

def test_optimization_results_saving(optimizer_config, temp_dir, sample_data):
    """Test optimization results saving and loading."""
    optimizer = StrategyOptimizer(
        strategy_class=MockStrategy,
        param_ranges=optimizer_config['param_ranges'],
        base_config=optimizer_config['base_config'],
        output_dir=str(temp_dir)
    )
    
    # Write sample data
    data_path = temp_dir / 'test_data.csv'
    sample_data.to_csv(data_path, index=False)
    optimizer.base_config['data_path'] = str(data_path)
    
    with patch('src.optimization.task_coordinator.TaskCoordinator.run_optimization') as mock_run:
        mock_run.return_value = {
            'summary': {'total_combinations': 10},
            'best_results': {'sharpe_ratio': {'value': 1.5, 'parameters': {'threshold': 1.0}}},
            'metrics': {'sharpe_ratio': {'mean': 1.0}},
            'results': [{'parameters': {'threshold': 1.0}, 'metrics': {'sharpe_ratio': 1.5}}]
        }
        
        results = optimizer.run_optimization(max_workers=1)
        
        assert isinstance(results, OptimizationResults)
        assert Path(results.report_path).exists()
        assert results.best_params['sharpe_ratio']['value'] == 1.5

def test_error_handling(temp_dir):
    """Test error handling in optimization process."""
    with pytest.raises(ValueError):
        StrategyOptimizer(
            strategy_class=MockStrategy,
            param_ranges={},  # Empty param ranges
            base_config={'data_path': 'test.csv'},
            output_dir=str(temp_dir)
        )

def test_optimization_progress_tracking(optimizer_config):
    """Test optimization progress tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        coordinator = TaskCoordinator(
            OptimizationConfig(
                strategy_class=MockStrategy,
                param_ranges=optimizer_config['param_ranges'],
                data_path='test.csv',
                base_config=optimizer_config['base_config'],
                output_dir=Path(tmpdir)
            )
        )
        
        progress = coordinator.get_progress_report()
        assert 'total_combinations' in progress
        assert 'progress_percentage' in progress
        assert 'worker_status' in progress

def test_parameter_grid_generation(optimizer_config):
    """Test parameter grid generation."""
    from src.optimization.parameter_grid import ParameterGrid
    
    grid = ParameterGrid(optimizer_config['param_ranges'])
    combinations = grid.generate_combinations()
    
    assert len(combinations) > 0
    assert all('threshold' in c and 'amount' in c for c in combinations)

def test_integration_flow():
    """Test complete optimization flow integration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        param_ranges = {
            'threshold': {'start': 1.0, 'end': 1.1, 'step': 0.1}
        }
        base_config = {
            'data_path': os.path.join(tmpdir, 'test_data.csv'),
            'asset': 'EUTEUR',
            'base_currency': 'EUR',
            'initial_cash': 1000
        }
        
        # Create test data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', '2023-01-02', freq='1H'),
            'close': np.random.uniform(0.99, 1.01, 25)
        })
        sample_data.to_csv(base_config['data_path'], index=False)
        
        # Initialize optimizer
        optimizer = StrategyOptimizer(
            strategy_class=MockStrategy,
            param_ranges=param_ranges,
            base_config=base_config,
            output_dir=tmpdir
        )
        
        # Run optimization with minimal settings
        with patch('src.optimization.celery_tasks.run_optimization.delay') as mock_task:
            mock_task.return_value.get.return_value = {
                'parameters': {'threshold': 1.0},
                'metrics': {'sharpe_ratio': 1.5}
            }
            
            results = optimizer.run_optimization(max_workers=1)
            
            assert results.summary['completed_tasks'] > 0
            assert Path(results.report_path).exists()
            assert Path(results.plots_directory).exists()