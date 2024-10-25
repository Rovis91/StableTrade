"""Data validation utilities for the visualization package."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Type, Set

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data before visualization."""
    
    # Class-level constants for required columns and their types
    REQUIRED_COLUMNS: Dict[str, Type] = {
        'depeg_threshold': np.number,
        'trade_amount': np.number,
        'stop_loss': np.number,
        'take_profit': np.number,
        'sharpe_ratio': np.number,
        'total_return': np.number,
        'max_drawdown': np.number,
        'win_rate': np.number
    }
    
    @classmethod
    def validate_results_data(cls, df: pd.DataFrame) -> bool:
        """
        Validate required columns and data types.
        
        Args:
            df: DataFrame containing optimization results
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValueError: If required columns are missing or data types are incorrect
            TypeError: If input is not a pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
            
        # Check missing columns
        missing_cols = set(cls.REQUIRED_COLUMNS.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Check data types
        for col, dtype in cls.REQUIRED_COLUMNS.items():
            if not np.issubdtype(df[col].dtype, dtype):
                raise ValueError(
                    f"Column '{col}' has incorrect type. "
                    f"Expected {dtype.__name__}, got {df[col].dtype.name}"
                )
            
            # Check for NaN values
            if df[col].isna().any():
                raise ValueError(f"Column '{col}' contains NaN values")
        
        logger.info("Data validation successful")
        return True

    @staticmethod
    def get_available_columns(df: pd.DataFrame) -> Set[str]:
        """Return set of available columns in DataFrame."""
        return set(df.columns)

    @staticmethod
    def check_numeric_columns(df: pd.DataFrame, columns: Set[str]) -> bool:
        """Check if specified columns are numeric."""
        for col in columns:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                return False
        return True