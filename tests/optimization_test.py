import os
import sys
import pytest
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimization.parameter_grid import ParameterGrid, ParameterRange


@pytest.fixture
def valid_param_ranges():
    """Fixture for valid parameter ranges."""
    return {
        'depeg_threshold': {'start': 1.0, 'end': 3.0, 'step': 0.5},
        'trade_amount': {'start': 0.1, 'end': 0.3, 'step': 0.1}
    }

@pytest.fixture
def parameter_grid(valid_param_ranges):
    """Fixture for parameter grid instance."""
    return ParameterGrid(valid_param_ranges)

def test_initialization_valid_params(valid_param_ranges):
    """Test initialization with valid parameters."""
    grid = ParameterGrid(valid_param_ranges)
    
    assert len(grid.param_ranges) == 2, "There should be two parameters initialized."
    assert isinstance(grid.param_ranges['depeg_threshold'], ParameterRange), "depeg_threshold should be of type ParameterRange."
    assert isinstance(grid.param_ranges['trade_amount'], ParameterRange), "trade_amount should be of type ParameterRange."
    assert grid.total_combinations == 15, "Total combinations should be 15 (5 values for depeg_threshold * 3 for trade_amount)."

def test_parameter_validation():
    """Test parameter validation with invalid values."""
    invalid_ranges = [
        # Negative step
        {
            'param': {'start': 1.0, 'end': 3.0, 'step': -0.5}
        },
        # End less than start
        {
            'param': {'start': 3.0, 'end': 1.0, 'step': 0.5}
        },
        # Non-numeric values
        {
            'param': {'start': 'invalid', 'end': '3.0', 'step': '0.5'}
        }
    ]
    
    for invalid_range in invalid_ranges:
        with pytest.raises(ValueError, match="Step size must be positive|End value must be greater than or equal to start value|could not convert string to float"):
            ParameterGrid(invalid_range)

def test_edge_cases():
    """Test edge cases in parameter ranges."""
    edge_cases = [
        # Zero step
        {
            'zero_step': {'start': 1.0, 'end': 3.0, 'step': 0.0}
        },
        # Step larger than range
        {
            'large_step': {'start': 1.0, 'end': 1.1, 'step': 0.5}
        }
    ]
    
    for case in edge_cases:
        with pytest.raises(ValueError, match="Step size must be positive|Step size is larger than the range"):
            ParameterGrid(case)

def test_minimum_valid_range():
    """Test minimum valid parameter range."""
    param_ranges = {
        'min_range': {'start': 1.0, 'end': 1.1, 'step': 0.1}
    }
    grid = ParameterGrid(param_ranges)
    combinations = grid.generate_combinations()
    
    assert len(combinations) == 2, "Combinations should include both start and end values."
    assert combinations[0]['min_range'] == 1.0, "First value should be the start of the range."
    assert combinations[1]['min_range'] == 1.1, "Second value should be the end of the range."

def test_logging_invalid_range(caplog):
    """Test logging of errors for invalid ranges."""
    invalid_range = {
        'param': {'start': 1.0, 'end': 3.0, 'step': -0.5}
    }
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            ParameterGrid(invalid_range)
    
    assert "Step size must be positive" in caplog.text, "Error message should be logged."

def test_large_parameter_combinations():
    """Test grid generation with a large number of combinations."""
    large_ranges = {
        'param1': {'start': 1.0, 'end': 10.0, 'step': 1.0},
        'param2': {'start': 0.1, 'end': 1.0, 'step': 0.1}
    }
    
    grid = ParameterGrid(large_ranges)
    combinations = grid.generate_combinations()
    
    assert len(combinations) == 100, "There should be 100 combinations for 10 * 10 grid."

def test_small_step_edge_case():
    """Test small step size close to floating point precision."""
    param_ranges = {
        'small_step': {'start': 1.0, 'end': 1.0000001, 'step': 0.00000005}
    }
    grid = ParameterGrid(param_ranges)
    combinations = grid.generate_combinations()
    
    assert len(combinations) == 3, "There should be 3 combinations even with small step size."
    assert combinations[0]['small_step'] == 1.0
    assert combinations[1]['small_step'] > 1.0

def test_zero_range():
    """Test when start and end of the range are the same."""
    param_ranges = {
        'zero_range': {'start': 1.0, 'end': 1.0, 'step': 0.1}
    }
    grid = ParameterGrid(param_ranges)
    combinations = grid.generate_combinations()
    
    assert len(combinations) == 1, "There should be only 1 combination when start and end are the same."
    assert combinations[0]['zero_range'] == 1.0
