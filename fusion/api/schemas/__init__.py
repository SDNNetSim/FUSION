"""
Pydantic schemas for request/response validation.

Defines API contracts for runs, configs, topology, and common types.
"""

from .common import ErrorResponse, PaginatedResponse
from .config import ConfigSchema, ConfigValidationResponse, TemplateInfo
from .run import RunCreate, RunListResponse, RunProgress, RunResponse
from .topology import (
    TopologyLink,
    TopologyListItem,
    TopologyListResponse,
    TopologyNode,
    TopologyResponse,
)

__all__ = [
    # Run schemas
    "RunCreate",
    "RunResponse",
    "RunListResponse",
    "RunProgress",
    # Config schemas
    "TemplateInfo",
    "ConfigSchema",
    "ConfigValidationResponse",
    # Topology schemas
    "TopologyNode",
    "TopologyLink",
    "TopologyResponse",
    "TopologyListItem",
    "TopologyListResponse",
    # Common schemas
    "ErrorResponse",
    "PaginatedResponse",
]
