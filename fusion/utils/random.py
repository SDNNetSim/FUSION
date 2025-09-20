"""Random number generation utilities for FUSION simulations."""

from typing import Optional, Union

import numpy as np


def set_random_seed(seed_value: int) -> None:
    """Set the seed for random number generation functions.
    
    Configures the numpy random number generator with the specified seed
    to ensure reproducible simulation results.
    
    :param seed_value: The seed value for random number generation
    :type seed_value: int
    :raises ValueError: If seed_value is negative
    
    Example:
        >>> set_random_seed(42)
        # All subsequent random number generation will be deterministic
    """
    if seed_value < 0:
        raise ValueError("Seed value must be non-negative")
    np.random.seed(seed_value)


def generate_uniform_random_variable(scale_parameter: Optional[float] = None) -> Union[float, int]:
    """Generate a value from a uniform distribution.
    
    Returns a random value uniformly distributed between 0 and 1 if no scale
    parameter is provided. If a scale parameter is given, returns an integer
    uniformly distributed between 0 and scale_parameter.
    
    :param scale_parameter: Optional scale parameter for integer generation
    :type scale_parameter: Optional[float]
    :return: Uniform random variable (float if unscaled, int if scaled)
    :rtype: Union[float, int]
    :raises ValueError: If scale_parameter is negative or zero
    
    Example:
        >>> random_float = generate_uniform_random_variable()
        >>> random_int = generate_uniform_random_variable(10.0)
    """
    if scale_parameter is None:
        return np.random.uniform(0, 1)

    if scale_parameter <= 0:
        raise ValueError("Scale parameter must be positive")

    return int(np.random.uniform(0, 1) * scale_parameter)


def generate_exponential_random_variable(scale_parameter: float) -> float:
    """Generate a value from an exponential distribution.
    
    Uses the inverse transform method to generate exponentially distributed
    random variables with the specified scale parameter (rate = 1/scale).
    
    :param scale_parameter: Scale parameter for the exponential distribution
    :type scale_parameter: float
    :return: Exponentially distributed random variable
    :rtype: float
    :raises ValueError: If scale_parameter is negative or zero
    
    Example:
        >>> arrival_time = generate_exponential_random_variable(2.5)
        # Generates exponential random variable with mean 2.5
    """
    if scale_parameter <= 0:
        raise ValueError("Scale parameter must be positive")

    # Use inverse transform method: F^(-1)(U) = -λ * ln(U)
    uniform_sample = generate_uniform_random_variable()
    return (-1.0 / scale_parameter) * np.log(uniform_sample)
