"""
Parameter grid generation module for optimization strategies.

This module provides utilities for generating and managing parameter combinations
for strategy optimization, with memory-efficient grid generation and validation.
"""

from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from typing import Dict, Any, Final, Iterator, Union
import logging
from functools import cached_property

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# Type aliases
NumericType = Union[int, float]
ParamDict = Dict[str, Dict[str, float]]
ParamInfo = Dict[str, Dict[str, Any]]
ParamCombination = Dict[str, float]

# Constants
PRECISION: Final[int] = 8
MIN_STEP_SIZE: Final[float] = 1e-10

@dataclass(frozen=True)
class ParameterRange:
    """
    Immutable parameter range specification.
    
    Attributes:
        start: Starting value of the range
        end: Ending value of the range
        step: Step size between values
        
    Raises:
        ValueError: If range values are invalid
    """
    start: float
    end: float
    step: float

    def __post_init__(self) -> None:
        """Validate range values after initialization."""
        self._validate_types()
        self._validate_values()

    def _validate_types(self) -> None:
        """Validate parameter types."""
        if not all(isinstance(x, (int, float)) for x in (self.start, self.end, self.step)):
            raise ValueError("All range values must be numeric")

    def _validate_values(self) -> None:
        """Validate parameter values."""
        if self.step <= MIN_STEP_SIZE:
            raise ValueError(f"Step size must be greater than {MIN_STEP_SIZE}")
        if self.end < self.start:
            raise ValueError("End value must be greater than or equal to start value")
        if self.end != self.start and abs(self.end - self.start) < self.step:
            raise ValueError("Step size cannot be larger than range")

    @cached_property
    def values(self) -> NDArray[np.float64]:
        """Generate array of parameter values."""
        return np.arange(
            self.start,
            self.end + self.step/2,
            self.step
        ).round(PRECISION)

    @property
    def n_steps(self) -> int:
        """Calculate number of steps in range."""
        return len(self.values)

class ParameterGrid:
    """
    Memory-efficient parameter grid generator for optimization.
    
    This class handles the generation and validation of parameter combinations,
    with emphasis on memory efficiency for large parameter spaces.
    """

    def __init__(self, param_ranges: ParamDict) -> None:
        """
        Initialize parameter grid generator.

        Args:
            param_ranges: Dictionary of parameter ranges with format:
                {
                    'param_name': {
                        'start': float,
                        'end': float,
                        'step': float
                    }
                }

        Raises:
            ValueError: If parameters are invalid
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not param_ranges:
            raise ValueError("Parameter ranges cannot be empty")
            
        self.param_ranges = self._convert_to_parameter_ranges(param_ranges)
        self.logger.info("Initialized ParameterGrid with %d parameters", len(param_ranges))

    def _convert_to_parameter_ranges(self, param_ranges: ParamDict) -> Dict[str, ParameterRange]:
        """Convert input dictionary to ParameterRange objects."""
        converted = {}
        
        for param_name, range_dict in param_ranges.items():
            try:
                required_keys = {'start', 'end', 'step'}
                missing_keys = required_keys - set(range_dict.keys())
                if missing_keys:
                    raise KeyError(f"Missing required keys: {missing_keys}")

                converted[param_name] = ParameterRange(
                    start=float(range_dict['start']),
                    end=float(range_dict['end']),
                    step=float(range_dict['step'])
                )
                self.logger.debug("Converted range for parameter: %s", param_name)
                
            except Exception as e:
                self.logger.error("Invalid range for %s: %s", param_name, str(e))
                raise ValueError(f"Invalid range for {param_name}: {str(e)}") from e

        return converted

    @cached_property
    def total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        total = 1
        for param_range in self.param_ranges.values():
            total *= param_range.n_steps
        return total

    def generate_combinations(self) -> Iterator[ParamCombination]:
        """
        Generate parameter combinations efficiently.
        
        Yields:
            Dictionary containing one parameter combination
            
        Note:
            Uses iterator to minimize memory usage
        """
        if not self.param_ranges:
            self.logger.warning("No parameter ranges defined")
            return

        param_names = list(self.param_ranges.keys())
        param_values = [
            self.param_ranges[name].values 
            for name in param_names
        ]
        
        self.logger.info("Generating %d parameter combinations", self.total_combinations)
        
        # Use meshgrid for efficient combination generation
        mesh = np.meshgrid(*param_values)
        for idx in range(self.total_combinations):
            yield {
                param_name: float(mesh[i].flat[idx])
                for i, param_name in enumerate(param_names)
            }

    def get_param_info(self) -> ParamInfo:
        """Get detailed parameter range information."""
        return {
            param_name: {
                'start': param_range.start,
                'end': param_range.end,
                'step': param_range.step,
                'n_steps': param_range.n_steps,
                'values': param_range.values.tolist()
            }
            for param_name, param_range in self.param_ranges.items()
        }

    def estimate_runtime(self, time_per_run: float) -> Dict[str, float]:
        """
        Estimate total runtime for all combinations.
        
        Args:
            time_per_run: Time in seconds for single parameter evaluation
            
        Returns:
            Dictionary with time estimates in different units
        """
        if time_per_run <= 0:
            raise ValueError("time_per_run must be positive")
            
        total_time = self.total_combinations * time_per_run
        return {
            'seconds': total_time,
            'minutes': total_time / 60,
            'hours': total_time / 3600
        }

    def __str__(self) -> str:
        """Generate human-readable representation."""
        params = [
            f"{name}: {range.start} to {range.end} (step: {range.step})"
            for name, range in self.param_ranges.items()
        ]
        return (
            f"ParameterGrid with {len(self.param_ranges)} parameters:\n"
            f"{chr(10).join(params)}\n"
            f"Total combinations: {self.total_combinations}"
        )