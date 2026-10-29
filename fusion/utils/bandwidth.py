"""Bandwidth parsing that preserves fractional Gbps while keeping legacy integer keys."""

import math
from typing import Any


def parse_bandwidth(value: Any) -> int | float:
    """Parse a numeric bandwidth without truncating fractional traffic demands."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Bandwidth must be finite")
    return int(number) if number.is_integer() else number


def bandwidth_key(value: Any) -> str:
    """Canonical modulation/statistics key: 100.0 -> '100', 44.5 -> '44.5'."""
    return str(parse_bandwidth(value))
