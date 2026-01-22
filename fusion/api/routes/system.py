"""
System endpoints for health checks and API information.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    """
    Health check endpoint.

    Returns basic health status for monitoring and load balancers.
    """
    return {"status": "healthy"}


@router.get("/version")
def get_version() -> dict:
    """
    Get API version information.

    Returns the current API version and FUSION version.
    """
    try:
        from fusion import __version__ as fusion_version  # type: ignore[attr-defined]
    except ImportError:
        fusion_version = "unknown"

    return {
        "api_version": "1.0.0",
        "fusion_version": fusion_version,
    }
