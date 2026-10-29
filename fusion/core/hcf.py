"""Immutable records for the HCF SDN allocation policy.

Spectrum search lives in SpectrumAssignment; SDNController manages resources.
This module contains records, power-unit conversion and bounded numeric reuse. Physical calculations remain in the existing
SnrMeasurements.check_gsnr_hcf and calculate_hcf_snr_link methods.

"""

import math
import pickle
import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

Amplifier = tuple[Any, Any, int, str]


@dataclass(frozen=True)
class HCFMode:
    """Explicit net rate, symbol rate, and GSNR threshold of one optical carrier."""

    name: str
    net_rate_gbps: float
    baud_rate_gbaud: float
    threshold_db: float
    roll_off: float = 0.0

    def __post_init__(self) -> None:
        values = (self.net_rate_gbps, self.baud_rate_gbaud, self.threshold_db, self.roll_off)
        if not self.name or not all(math.isfinite(v) for v in values):
            raise ValueError("HCF mode values must be finite and name must be nonempty")
        if min(self.net_rate_gbps, self.baud_rate_gbaud) <= 0 or not 0 <= self.roll_off <= 1:
            raise ValueError("HCF rates must be positive and roll_off must be in [0, 1]")


@dataclass(frozen=True)
class HCFCandidate:
    """A single channel; demands contain (amplifier ID, reserved watts, budget watts)."""

    path: tuple[Any, ...]
    band: str
    channel: int
    mode: HCFMode
    power_w: float
    gsnr_db: float
    demands: tuple[tuple[Amplifier, float, float], ...]


@dataclass(frozen=True)
class HCFSpectrumResult:
    """One path's search result, returned without reserving spectrum or power."""

    candidate: HCFCandidate | None
    score: float
    free_channels: int
    qot_feasible_channels: int
    power_rejections: int


def dbm_to_watts(dbm: float) -> float:
    """Convert finite dBm to strictly positive watts."""
    if not math.isfinite(dbm):
        raise ValueError("HCF power must be finite")
    value = 10 ** ((dbm - 30) / 10)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("HCF power is outside the supported numeric range")
    return value


def hcf_power_objective(props: Mapping[str, Any]) -> str:
    """Resolve an explicit objective, retaining the legacy boolean when omitted."""
    objective = props.get("hcf_power_objective")
    if objective is None:
        return "max_gsnr" if props.get("optimized_power", False) else "discrete"
    if objective not in ("fixed", "discrete", "max_gsnr", "minimum_required"):
        raise ValueError("hcf_power_objective must be fixed, discrete, max_gsnr or minimum_required")
    return objective

class HCFPhysicalCache:
    """Bounded numeric reuse, never candidates, demands, occupancy or scores.

    Serialized keys contain values, not references to the network. Only dumps
    is used; nothing is deserialized. Accounted bytes include key/value sizes
    and a conservative per-entry allowance, not whole-process RSS.
    """

    def __init__(self, max_entries: int = 16384, max_bytes: int = 16 * 1024 * 1024) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[bytes, tuple[tuple[float, ...] | None, int]] = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(*values: Any) -> bytes:
        """Make an immutable value key, including all physical inputs."""
        return pickle.dumps(values, protocol=5)

    def get(self, key: bytes) -> tuple[bool, tuple[float, ...] | None]:
        if key not in self._entries:
            self.misses += 1
            return False, None
        self.hits += 1
        self._entries.move_to_end(key)
        return True, self._entries[key][0]

    def put(self, key: bytes, value: tuple[float, ...] | None) -> None:
        size = sys.getsizeof(key) + sys.getsizeof(value) + 256
        if value is not None:
            size += sum(sys.getsizeof(item) for item in value)
        if self.max_entries <= 0 or size > self.max_bytes:
            return
        if key in self._entries:
            self.current_bytes -= self._entries.pop(key)[1]
        while self._entries and (len(self._entries) >= self.max_entries or self.current_bytes + size > self.max_bytes):
            self.current_bytes -= self._entries.popitem(last=False)[1][1]
        self._entries[key] = (value, size)
        self.current_bytes += size

    def clear(self) -> None:
        self._entries.clear()
        self.current_bytes = self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)
