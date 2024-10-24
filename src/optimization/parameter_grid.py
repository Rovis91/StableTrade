import numpy as np
from typing import Dict, List, Any
import logging
from dataclasses import dataclass

# Configure module logger
logger = logging.getLogger(__name__)

@dataclass
class ParameterRange:
    """
    Data class to store parameter range information.

    Attributes:
        start (float): Starting value of the parameter range.
        end (float): Ending value of the parameter range.
        step (float): Step size between values.
    """
    start: float
    end: float
    step: float

    def __post_init__(self):
        """Validate parameter range values after initialization."""
        if not all(isinstance(x, (int, float)) for x in [self.start, self.end, self.step]):
            raise ValueError("All values must be numeric")
        if self.step <= 0:
            raise ValueError("Step size must be positive")
        if self.end < self.start:
            raise ValueError("End value must be greater than or equal to start value")
        if self.end != self.start and abs(self.end - self.start) < self.step:
            raise ValueError("Step size is larger than the range")


class ParameterGrid:
    """
    Generates and manages parameter combinations for strategy optimization.

    Attributes:
        param_ranges (Dict[str, ParameterRange]): Dictionary of parameter ranges.
        total_combinations (int): Total number of possible parameter combinations.
    """

    def __init__(self, param_ranges: Dict[str, Dict[str, float]]):
        """
        Initialize the parameter grid generator.

        Args:
            param_ranges (Dict[str, Dict[str, float]]): Dictionary of parameter ranges with format:
                {
                    'param_name': {'start': float, 'end': float, 'step': float}
                }

        Raises:
            ValueError: If parameter ranges are invalid or incorrectly formatted.
        """
        self.logger = logger.getChild(self.__class__.__name__)
        self.param_ranges = self._convert_to_parameter_ranges(param_ranges)
        self.total_combinations = self._calculate_total_combinations()
        self.logger.info(f"Initialized ParameterGrid with {len(param_ranges)} parameters")

    def _validate_range_values(self, param_name: str, start: float, end: float, step: float) -> None:
        """Validate parameter range values."""
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not isinstance(step, (int, float)):
            raise ValueError(f"Parameter {param_name}: All values must be numeric")
        if step <= 0:
            raise ValueError(f"Parameter {param_name}: Step size must be positive")
        if end < start:
            raise ValueError(f"Parameter {param_name}: End value must be greater than or equal to start value")
        if abs(end - start) < step and end != start:
            raise ValueError(f"Parameter {param_name}: Step size is larger than the range")

    def _convert_to_parameter_ranges(self, param_ranges: Dict[str, Dict[str, float]]) -> Dict[str, ParameterRange]:
        """
        Convert input dictionary to ParameterRange objects.

        Args:
            param_ranges (Dict[str, Dict[str, float]]): Dictionary of parameter ranges.

        Returns:
            Dict[str, ParameterRange]: Dictionary mapping parameter names to ParameterRange objects.
        """
        converted = {}
        for param_name, range_dict in param_ranges.items():
            try:
                start, end, step = float(range_dict['start']), float(range_dict['end']), float(range_dict['step'])
                self._validate_range_values(param_name, start, end, step)
                converted[param_name] = ParameterRange(start=start, end=end, step=step)
                self.logger.debug(f"Successfully converted range for parameter: {param_name}")
            except KeyError as e:
                self.logger.error(f"Missing required key for {param_name}: {e}")
                raise ValueError(f"Invalid parameter range format for {param_name}: missing {e}")
            except (ValueError, TypeError) as e:
                self.logger.error(f"Invalid parameter range for {param_name}: {str(e)}")
                raise ValueError(f"Invalid parameter range format for {param_name}: {str(e)}")

        return converted

    def _calculate_total_combinations(self) -> int:
        """
        Calculate the total number of parameter combinations.

        Returns:
            int: Total number of possible parameter combinations.
        """
        total = 1
        for param_range in self.param_ranges.values():
            n_steps = int(np.ceil((param_range.end - param_range.start) / param_range.step)) + 1
            total *= n_steps
        return total

    def generate_combinations(self) -> List[Dict[str, float]]:
        """
        Generate all possible parameter combinations.

        Returns:
            List[Dict[str, float]]: List of dictionaries, where each dictionary represents one parameter combination.
        """
        if not self.param_ranges:
            self.logger.warning("No parameter ranges defined")
            return []

        param_values = {
            param_name: np.arange(
                param_range.start, param_range.end + param_range.step / 2, param_range.step
            ).round(8) for param_name, param_range in self.param_ranges.items()
        }

        param_names = list(self.param_ranges.keys())
        mesh = np.meshgrid(*[param_values[param] for param in param_names])

        combinations = [
            {param_name: float(mesh[i].flat[idx]) for i, param_name in enumerate(param_names)}
            for idx in range(len(mesh[0].flat))
        ]

        self.logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations

    def estimate_calculation_time(self, single_run_time: float) -> float:
        """
        Estimate total calculation time in seconds.

        Args:
            single_run_time (float): Estimated time for a single backtest run in seconds.

        Returns:
            float: Estimated total time in seconds.
        """
        return self.total_combinations * single_run_time

    def get_param_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about parameter ranges.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary containing detailed information about each parameter range.
        """
        return {
            param_name: {
                'start': param_range.start,
                'end': param_range.end,
                'step': param_range.step,
                'n_steps': int(np.ceil((param_range.end - param_range.start) / param_range.step)) + 1,
                'values': np.arange(
                    param_range.start, param_range.end + param_range.step / 2, param_range.step
                ).round(8).tolist()
            } for param_name, param_range in self.param_ranges.items()
        }

    def __str__(self) -> str:
        """
        Generate string representation of the ParameterGrid.

        Returns:
            str: String showing parameter ranges and total combinations.
        """
        params_str = "\n".join(
            f"{param_name}: {param_range.start} to {param_range.end} (step: {param_range.step})"
            for param_name, param_range in self.param_ranges.items()
        )
        return f"ParameterGrid with {len(self.param_ranges)} parameters:\n{params_str}\nTotal combinations: {self.total_combinations}"
