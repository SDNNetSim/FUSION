"""Value object for measured metric values."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict
from enum import Enum
import numpy as np


class DataType(Enum):
    """Supported data types for metrics."""

    FLOAT = "float"
    INT = "int"
    ARRAY = "array"
    DICT = "dict"
    STRING = "string"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class MetricValue:
    """
    Immutable container for a measured metric value.

    Attributes:
        value: The actual measured value
        data_type: Type of the data
        unit: Optional unit of measurement
        metadata: Optional additional metadata
    """

    value: Any
    data_type: DataType
    unit: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate the metric value."""
        if self.metadata is None:
            # Use object.__setattr__ since dataclass is frozen
            object.__setattr__(self, 'metadata', {})

    @property
    def as_float(self) -> float:
        """Return value as float."""
        if self.data_type == DataType.FLOAT:
            return float(self.value)
        elif self.data_type == DataType.INT:
            return float(self.value)
        else:
            raise TypeError(f"Cannot convert {self.data_type} to float")

    @property
    def as_int(self) -> int:
        """Return value as int."""
        if self.data_type == DataType.INT:
            return int(self.value)
        elif self.data_type == DataType.FLOAT:
            return int(self.value)
        else:
            raise TypeError(f"Cannot convert {self.data_type} to int")

    @property
    def as_array(self) -> np.ndarray:
        """Return value as numpy array."""
        if self.data_type == DataType.ARRAY:
            if isinstance(self.value, np.ndarray):
                return self.value
            return np.array(self.value)
        else:
            raise TypeError(f"Cannot convert {self.data_type} to array")

    @property
    def as_dict(self) -> Dict[str, Any]:
        """Return value as dictionary."""
        if self.data_type == DataType.DICT:
            return dict(self.value)
        else:
            raise TypeError(f"Cannot convert {self.data_type} to dict")

    def __repr__(self) -> str:
        """Return detailed representation."""
        unit_str = f" {self.unit}" if self.unit else ""
        return f"MetricValue({self.value}{unit_str}, type={self.data_type.value})"
