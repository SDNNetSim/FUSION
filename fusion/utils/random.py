import numpy as np


def set_seed(seed: int):
    """
    Sets the seed for random number generation functions.

    :param seed: The seed
    """
    np.random.seed(seed)


def get_uniform_rv(scale_param: float = None):
    """
    Generates a value from a uniform distribution. Optional scale parameter.

    :param scale_param: A scale parameter
    :return: A uniform random variable
    :rtype: int
    """
    if scale_param is None:
        return np.random.uniform(0, 1)

    return int(np.random.uniform(0, 1) * scale_param)


def get_exponential_rv(scale_param: float):
    """
    Generates a value from an exponential distribution.

    :param scale_param: A scale parameter
    :return: An exponential random variable
    :rtype: float
    """
    # np.log is the natural logarithm
    return ((-1.0) / float(scale_param)) * np.log(get_uniform_rv())
