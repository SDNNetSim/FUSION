"""
Run management endpoints.

Provides CRUD operations for simulation runs and log streaming.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ..db.database import get_db
from ..db.models import Run
from ..schemas.run import RunCreate, RunListResponse, RunProgress, RunResponse
from ..services.progress_watcher import stream_progress
from ..services.run_manager import RunManager, stream_run_logs

# Type alias for dependency injection
DBSession = Annotated[Session, Depends(get_db)]

router = APIRouter()


def _run_to_response(run: Run) -> RunResponse:
    """
    Convert a Run ORM object to a RunResponse schema.

    :param run: The Run ORM object.
    :returns: The corresponding RunResponse schema.
    """
    progress = None
    if run.status == "RUNNING":
        progress = RunProgress(
            current_erlang=run.current_erlang,
            total_erlangs=run.total_erlangs,
            current_iteration=run.current_iteration,
            total_iterations=run.total_iterations,
            percent_complete=(
                (run.current_iteration / run.total_iterations * 100) if run.total_iterations and run.current_iteration else None
            ),
        )

    return RunResponse(
        id=run.id,
        name=run.name,
        status=run.status,
        template=run.template,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        error_message=run.error_message,
        progress=progress,
    )


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(
    data: RunCreate,
    db: DBSession,
) -> RunResponse:
    """
    Create and start a new simulation run.

    :param data: Run configuration.
    :param db: Database session.
    :returns: The created run.
    """
    manager = RunManager(db)
    try:
        run = manager.create_run(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return _run_to_response(run)


@router.get("", response_model=RunListResponse)
def list_runs(
    db: DBSession,
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> RunListResponse:
    """
    List all runs with optional filtering.

    :param status_filter: Comma-separated list of statuses to filter by.
    :param limit: Maximum number of runs to return.
    :param offset: Number of runs to skip.
    :param db: Database session.
    :returns: Paginated list of runs.
    """
    query = db.query(Run).order_by(Run.created_at.desc())

    if status_filter:
        statuses = [s.strip().upper() for s in status_filter.split(",")]
        query = query.filter(Run.status.in_(statuses))

    total = query.count()
    runs = query.offset(offset).limit(limit).all()

    return RunListResponse(
        runs=[_run_to_response(r) for r in runs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{run_id}", response_model=RunResponse)
def get_run(run_id: str, db: DBSession) -> RunResponse:
    """
    Get details for a specific run.

    :param run_id: The run identifier.
    :param db: Database session.
    :returns: The run details.
    """
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return _run_to_response(run)


@router.delete("/{run_id}", response_model=RunResponse)
def cancel_run(run_id: str, db: DBSession) -> RunResponse:
    """
    Cancel a running job or delete a completed one.

    :param run_id: The run identifier.
    :param db: Database session.
    :returns: The cancelled/deleted run.
    """
    manager = RunManager(db)
    run = manager.cancel_or_delete(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return _run_to_response(run)


@router.get("/{run_id}/logs")
async def stream_logs(
    run_id: str,
    db: DBSession,
    from_start: bool = Query(True, description="Whether to send existing content"),
) -> EventSourceResponse:
    """
    Stream logs via Server-Sent Events.

    :param run_id: The run identifier.
    :param from_start: Whether to send existing log content first.
    :param db: Database session.
    :returns: SSE stream of log events.
    """
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return EventSourceResponse(stream_run_logs(run_id, from_start))


@router.get("/{run_id}/progress")
async def stream_run_progress(
    run_id: str,
    db: DBSession,
) -> EventSourceResponse:
    """
    Stream progress events via Server-Sent Events.

    :param run_id: The run identifier.
    :param db: Database session.
    :returns: SSE stream of progress events.
    """
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return EventSourceResponse(stream_progress(run_id))
