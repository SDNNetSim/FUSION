"""Pydantic schemas for topology API endpoints."""

from pydantic import BaseModel


class TopologyNode(BaseModel):
    """A node in the network topology."""

    id: str
    label: str
    x: float
    y: float
    type: str = "default"


class TopologyLink(BaseModel):
    """A link in the network topology."""

    id: str
    source: str
    target: str
    length_km: float
    utilization: float = 0.0


class TopologyResponse(BaseModel):
    """Response containing full topology data."""

    name: str
    nodes: list[TopologyNode]
    links: list[TopologyLink]


class TopologyListItem(BaseModel):
    """Summary info for a topology in the list."""

    name: str
    node_count: int
    link_count: int


class TopologyListResponse(BaseModel):
    """Response containing list of available topologies."""

    topologies: list[TopologyListItem]
