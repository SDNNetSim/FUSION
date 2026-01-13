"""
Failure type registry for dynamic handler lookup.
"""

from collections.abc import Callable

from .errors import InvalidFailureTypeError
from .failure_types import fail_geo, fail_link, fail_node, fail_srlg

# Registry of failure handlers
FAILURE_TYPES: dict[str, Callable] = {
    "link": fail_link,
    "node": fail_node,
    "srlg": fail_srlg,
    "geo": fail_geo,
}


def get_failure_handler(failure_type: str) -> Callable:
    """
    Get failure handler function by type.

    :param failure_type: Failure type (link, node, srlg, geo)
    :type failure_type: str
    :return: Failure handler function
    :rtype: Callable
    :raises InvalidFailureTypeError: If failure type is unknown

    Example:
        >>> handler = get_failure_handler('link')
        >>> event = handler(topology, link_id=(0, 1), t_fail=10.0, t_repair=20.0)
    """
    if failure_type not in FAILURE_TYPES:
        raise InvalidFailureTypeError(
            f"Unknown failure type: {failure_type}. "
            f"Valid types: {list(FAILURE_TYPES.keys())}"
        )
    return FAILURE_TYPES[failure_type]


def register_failure_type(name: str, handler: Callable) -> None:
    """
    Register a custom failure type handler.

    Allows extending the failure module with custom failure types.

    :param name: Failure type name
    :type name: str
    :param handler: Failure handler function
    :type handler: Callable
    """
    FAILURE_TYPES[name] = handler
