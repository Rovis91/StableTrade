import pytest
from unittest.mock import patch, Mock
from src.optimization.celery_tasks import run_backtest, process_optimization_results, create_optimization_tasks, monitor_task_progress

# Fixtures for mock dependencies
@pytest.fixture
def base_config():
    """Fixture for base backtest configuration."""
    return {
        'base_currency': 'USD',
        'asset': 'BTCUSD',
        'initial_cash': 10000,
        'data_path': 'path/to/data.csv',
        'slippage': 0.001
    }

@pytest.fixture
def param_combination():
    """Fixture for a sample parameter combination."""
    return {
        'depeg_threshold': 1.5,
        'trade_amount': 0.2,
        'stop_loss': 0.1,
        'take_profit': 0.3
    }

@pytest.fixture
def param_combinations():
    """Fixture for a list of parameter combinations."""
    return [
        {'depeg_threshold': 1.0, 'trade_amount': 0.1},
        {'depeg_threshold': 1.5, 'trade_amount': 0.2},
        {'depeg_threshold': 2.0, 'trade_amount': 0.3}
    ]

# Test cases

def test_run_backtest_success(param_combination, base_config):
    """Test the successful execution of the run_backtest task."""
    with patch('src.optimization.celery_tasks.BacktestEngine') as MockBacktestEngine, \
         patch('src.optimization.celery_tasks.MetricsModule') as MockMetricsModule:
        
        mock_backtest_engine = MockBacktestEngine.return_value
        mock_metrics_module = MockMetricsModule.return_value
        
        mock_backtest_engine.run_backtest.return_value = None
        mock_metrics_module.get_summary.return_value = {'metric1': 0.5, 'metric2': 0.8}
        
        result = run_backtest(param_combination, base_config)
        
        assert 'parameters' in result
        assert 'metrics' in result
        assert 'execution_time' in result
        assert result['parameters'] == param_combination
        assert result['metrics'] == {'metric1': 0.5, 'metric2': 0.8}

def test_run_backtest_failure(param_combination, base_config):
    """Test the execution of the run_backtest task when an exception occurs."""
    with patch('src.optimization.celery_tasks.BacktestEngine', side_effect=Exception('Test Error')):
        result = run_backtest(param_combination, base_config)
        
        assert 'parameters' in result
        assert 'error' in result
        assert result['status'] == 'failed'

def test_process_optimization_results_success(param_combinations, tmp_path):
    """Test the successful execution of process_optimization_results."""
    results = [
        {'parameters': param, 'metrics': {'metric1': 0.5, 'metric2': 0.8}, 'execution_time': 10}
        for param in param_combinations
    ]
    
    summary = process_optimization_results(results, str(tmp_path))
    
    assert 'total_runs' in summary
    assert 'successful_runs' in summary
    assert 'failed_runs' in summary
    assert summary['total_runs'] == len(param_combinations)
    assert summary['successful_runs'] == len(param_combinations)
    assert summary['failed_runs'] == 0

def test_create_optimization_tasks(param_combinations, base_config, tmp_path):
    """Test the creation of optimization tasks group."""
    task_group = create_optimization_tasks(param_combinations, base_config, str(tmp_path))
    assert len(task_group.tasks) == len(param_combinations)

def test_create_optimization_tasks_empty(param_combinations, base_config, tmp_path):
    """Test creating optimization tasks with empty param_combinations."""
    empty_combinations = []
    task_group = create_optimization_tasks(empty_combinations, base_config, str(tmp_path))
    assert len(task_group.tasks) == 0

def test_monitor_task_progress():
    """Test monitoring the progress of tasks."""
    mock_task_group = Mock()
    mock_task_group.tasks = [Mock(status='SUCCESS'), Mock(status='SUCCESS'), Mock(status='FAILURE')]
    
    with patch('src.optimization.celery_tasks.time.sleep', return_value=None):
        with patch('src.optimization.celery_tasks.logger') as mock_logger:
            monitor_task_progress(mock_task_group)
            
            mock_logger.info.assert_any_call('Progress: 2/3 completed, 1 failed')
            mock_logger.info.assert_any_call('All tasks completed. Successful: 2, Failed: 1')

def test_monitor_task_progress_all_success():
    """Test monitoring task progress when all tasks are successful."""
    mock_task_group = Mock()
    mock_task_group.tasks = [Mock(status='SUCCESS'), Mock(status='SUCCESS')]
    
    with patch('src.optimization.celery_tasks.time.sleep', return_value=None):
        with patch('src.optimization.celery_tasks.logger') as mock_logger:
            monitor_task_progress(mock_task_group)
            
            mock_logger.info.assert_any_call('Progress: 2/2 completed, 0 failed')
            mock_logger.info.assert_any_call('All tasks completed. Successful: 2, Failed: 0')

def test_monitor_task_progress_all_failure():
    """Test monitoring task progress when all tasks fail."""
    mock_task_group = Mock()
    mock_task_group.tasks = [Mock(status='FAILURE'), Mock(status='FAILURE')]
    
    with patch('src.optimization.celery_tasks.time.sleep', return_value=None):
        with patch('src.optimization.celery_tasks.logger') as mock_logger:
            monitor_task_progress(mock_task_group)
            
            mock_logger.info.assert_any_call('Progress: 0/2 completed, 2 failed')
            mock_logger.info.assert_any_call('All tasks completed. Successful: 0, Failed: 2')
