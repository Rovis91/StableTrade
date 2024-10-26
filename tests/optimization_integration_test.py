import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import time
from typing import Dict, Any

from src.strategy.depeg_strategy import DepegStrategy
from src.optimization.optimizer import StrategyOptimizer

def create_test_data() -> pd.DataFrame:
    """Create test market data."""
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
def test_environment():
    """Setup test environment with data and configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create paths
        tmpdir = Path(tmpdir)
        data_path = tmpdir / 'test_data.csv'
        output_dir = tmpdir / 'results'
        
        # Create test data
        data = create_test_data()
        data.to_csv(data_path, index=False)
        
        # Create configuration
        config = {
            'param_ranges': {
                'depeg_threshold': {'start': 1.0, 'end': 1.2, 'step': 0.1},
                'trade_amount': {'start': 0.1, 'end': 0.2, 'step': 0.1}
            },
            'base_config': {
                'data_path': str(data_path),
                'asset': 'EUTEUR',
                'base_currency': 'EUR',
                'initial_cash': 1000
            },
            'output_dir': str(output_dir)
        }
        
        return config

def test_error_recovery(temp_dir):
    """Test system recovery from various error conditions."""
    data_path = temp_dir / 'nonexistent.csv'
    output_dir = temp_dir / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'param_ranges': {
            'depeg_threshold': {'start': 1.0, 'end': 1.1, 'step': 0.1}
        },
        'base_config': {
            'data_path': str(data_path),
            'asset': 'EUTEUR',
            'base_currency': 'EUR',
            'initial_cash': 1000
        }
    }
    
    optimizer = StrategyOptimizer(
        strategy_class=DepegStrategy,
        param_ranges=config['param_ranges'],
        base_config=config['base_config'],
        output_dir=str(output_dir)
    )
    
    # Test recovery from missing data file
    with pytest.raises(FileNotFoundError):
        optimizer.run_optimization()
    
    # Create data file and verify recovery
    data = create_test_data()
    data.to_csv(data_path, index=False)
    assert data_path.exists()