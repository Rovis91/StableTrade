import os
import sys
import pytest
import pandas as pd

# Add src path to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.result_manager import OptimizationResults


# ---------------------------- Fixtures ----------------------------

@pytest.fixture
def empty_results():
    """Fixture for an empty OptimizationResults instance."""
    return OptimizationResults()

@pytest.fixture
def sample_params():
    """Fixture for sample parameter values."""
    return {
        'depeg_threshold': 2.5,
        'trade_amount': 0.1,
        'stop_loss': 0.02
    }

@pytest.fixture
def sample_metrics():
    """Fixture for sample metric values."""
    return {
        'total_return': 15.5,
        'sharpe_ratio': 1.8,
        'max_drawdown': 0.12,
        'win_rate': 65.0
    }

@pytest.fixture
def populated_results(empty_results, sample_params, sample_metrics):
    """Fixture for OptimizationResults instance with sample data."""
    results = empty_results
    # Add multiple results with variations
    for i in range(5):
        modified_params = {k: v + i * 0.1 for k, v in sample_params.items()}
        modified_metrics = {k: v + i * 0.5 for k, v in sample_metrics.items()}
        results.add_result(modified_params, modified_metrics)
    return results


# ------------------------- Initialization Tests -------------------------

def test_initialization():
    """Test initialization of OptimizationResults."""
    results = OptimizationResults()
    assert results.results.empty
    assert results.param_columns == []
    assert results.metric_columns == []

def test_add_valid_result(empty_results, sample_params, sample_metrics):
    """Test adding a valid result."""
    empty_results.add_result(sample_params, sample_metrics)
    assert len(empty_results.results) == 1
    assert set(empty_results.param_columns) == set(sample_params.keys())
    assert set(empty_results.metric_columns) == set(sample_metrics.keys())

def test_add_invalid_result(empty_results, sample_params, sample_metrics):
    """Test adding invalid results."""
    # First add a valid result to establish structure
    empty_results.add_result(sample_params, sample_metrics)
    
    # Try adding result with missing parameter
    invalid_params = {k: v for k, v in sample_params.items() if k != 'depeg_threshold'}
    with pytest.raises(ValueError):
        empty_results.add_result(invalid_params, sample_metrics)
        
    # Try adding result with invalid value type
    invalid_params = {**sample_params, 'depeg_threshold': 'invalid'}
    with pytest.raises(ValueError):
        empty_results.add_result(invalid_params, sample_metrics)


# --------------------------- CSV Operations Tests ---------------------------

def test_save_and_load_csv(populated_results, tmp_path):
    """Test saving and loading results from CSV."""
    filepath = tmp_path / "test_results.csv"
    
    # Save results
    populated_results.save_to_csv(str(filepath))
    assert filepath.exists()
    
    # Load results into new instance
    new_results = OptimizationResults()
    new_results.load_from_csv(str(filepath))
    
    # Verify data matches - we need to sort columns first as order might differ
    pd.testing.assert_frame_equal(populated_results.results, new_results.results)
    
    # Compare columns sets rather than lists to ignore order
    assert set(populated_results.param_columns) == set(new_results.param_columns)
    assert set(populated_results.metric_columns) == set(new_results.metric_columns)

def test_save_empty_results(empty_results, tmp_path):
    """Test saving empty results."""
    filepath = tmp_path / "empty_results.csv"
    empty_results.save_to_csv(str(filepath))
    assert not filepath.exists()  # Should not create file for empty results


# ----------------------- Results and Statistics Tests -----------------------

def test_filter_results(populated_results):
    """Test filtering results based on conditions."""
    conditions = {
        'total_return': lambda x: x > 16.0,
        'sharpe_ratio': lambda x: x > 2.0
    }
    filtered = populated_results.filter_results(conditions)
    assert len(filtered) < len(populated_results.results)
    assert all(row['total_return'] > 16.0 and row['sharpe_ratio'] > 2.0 
              for _, row in filtered.iterrows())

def test_calculate_statistics(populated_results):
    """Test calculation of statistics."""
    stats = populated_results.calculate_statistics()
    
    # Check structure and basic statistical properties
    for column in populated_results.param_columns + populated_results.metric_columns:
        assert column in stats
        assert all(key in stats[column] for key in ['min', 'max', 'mean', 'std', 'median'])
        assert stats[column]['min'] <= stats[column]['max']
        assert stats[column]['min'] <= stats[column]['mean'] <= stats[column]['max']

def test_get_parameter_performance(populated_results):
    """Test getting performance statistics for a specific parameter."""
    analysis = populated_results.get_parameter_performance('depeg_threshold', 'total_return')
    
    # Check structure
    assert 'value_stats' in analysis
    assert 'best_value' in analysis
    assert 'worst_value' in analysis
    assert 'unique_values' in analysis
    
    # Check basic properties
    assert analysis['best_value'] >= analysis['worst_value']
    assert len(analysis['unique_values']) > 0

def test_get_top_results(populated_results):
    """Test getting top N results."""
    n = 3
    top_results = populated_results.get_top_results('total_return', n)
    
    assert len(top_results) == n
    assert top_results['total_return'].is_monotonic_decreasing  # Verify sorting

def test_compare_results(populated_results):
    """Test comparing two sets of results."""
    # Create another instance with slightly different results
    other_results = OptimizationResults()
    for i in range(5):
        params = {'depeg_threshold': 2.5 + i * 0.1, 
                 'trade_amount': 0.1 + i * 0.05, 
                 'stop_loss': 0.02 + i * 0.01}
        metrics = {'total_return': 14.0 + i * 0.5, 
                  'sharpe_ratio': 1.6 + i * 0.2,
                  'max_drawdown': 0.15 + i * 0.01,
                  'win_rate': 60.0 + i * 1.0}
        other_results.add_result(params, metrics)
    
    comparison = populated_results.compare_results(other_results)
    
    # Check structure and basic properties
    for metric in populated_results.metric_columns:
        assert metric in comparison
        assert all(key in comparison[metric] for key in 
                  ['this_mean', 'other_mean', 'difference_pct', 'this_best', 'other_best'])

def test_get_summary(populated_results):
    """Test getting summary of optimization results."""
    summary = populated_results.get_summary()
    
    # Check structure
    assert all(key in summary for key in 
              ['total_combinations', 'parameters_tested', 'metrics_evaluated', 
               'best_results', 'parameter_ranges'])
    
    # Check basic properties
    assert summary['total_combinations'] == len(populated_results.results)
    assert len(summary['parameters_tested']) == len(populated_results.param_columns)
    assert len(summary['metrics_evaluated']) == len(populated_results.metric_columns)


# ------------------------- Edge Case Tests -------------------------

def test_empty_operations(empty_results):
    """Test operations on empty results."""
    assert empty_results.calculate_statistics() == {}
    assert empty_results.calculate_correlations().empty
    assert empty_results.get_summary() == {}
    
    with pytest.raises(ValueError):
        empty_results.get_top_results('total_return')

def test_analyze_parameter_sensitivity(populated_results):
    """Test parameter sensitivity analysis."""
    analysis = populated_results.analyze_parameter_sensitivity('depeg_threshold', 'total_return')
    
    assert 'sensitivity_by_range' in analysis
    assert len(analysis['sensitivity_by_range']) > 0
    
    # Check structure of sensitivity analysis
    for entry in analysis['sensitivity_by_range']:
        assert all(key in entry for key in ['param_range', 'metric_change', 'sensitivity'])
        assert isinstance(entry['param_range'], tuple)
        assert len(entry['param_range']) == 2
        assert entry['param_range'][0] < entry['param_range'][1]

def test_invalid_metric_parameter_names(populated_results):
    """Test behavior with invalid metric or parameter names."""
    with pytest.raises(ValueError):
        populated_results.get_parameter_performance('invalid_param', 'total_return')
    
    with pytest.raises(ValueError):
        populated_results.get_parameter_performance('depeg_threshold', 'invalid_metric')
    
    with pytest.raises(ValueError):
        populated_results.get_top_results('invalid_metric')


# ----------------------- Correlation Test ------------------------

def test_calculate_correlations(populated_results):
    """Test correlation calculation."""
    import numpy as np  # Import np only in this test since it's specific here
    
    correlations = populated_results.calculate_correlations()
    
    # Check structure
    assert isinstance(correlations, pd.DataFrame)
    assert sorted(correlations.columns) == sorted(correlations.index)
    assert all(col in correlations.columns 
              for col in populated_results.param_columns + populated_results.metric_columns)
    
    # Check correlation matrix properties
    diag_vals = np.diag(correlations.values)
    np.testing.assert_allclose(diag_vals, np.ones_like(diag_vals), rtol=1e-15)  # Check diagonal is 1
    
    # Check all correlations are between -1 and 1 with tolerance
    np.testing.assert_allclose(
        correlations.values,
        np.clip(correlations.values, -1.0, 1.0),
        rtol=1e-15
    )
    
    # Verify symmetry of correlation matrix
    np.testing.assert_allclose(
        correlations.values,
        correlations.values.T,
        rtol=1e-15
    )
