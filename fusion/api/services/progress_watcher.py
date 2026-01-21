"""
Progress watcher service for monitoring simulation progress.

Watches progress.jsonl files and streams updates via SSE.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import aiofiles

from ..config import settings
from ..db.database import SessionLocal
from ..db.models import Run

logger = logging.getLogger(__name__)


async def stream_progress(run_id: str) -> AsyncGenerator[dict, None]:
    """
    Stream progress events from progress.jsonl via SSE.

    Watches the progress file for new events and yields them as SSE messages.
    Also updates the database with the latest progress.

    :param run_id: The run identifier.
    :yields: SSE event dictionaries with progress data.
    """
    progress_path = settings.runs_dir / run_id / "progress.jsonl"
    last_position = 0

    # Wait for file to exist (simulation might not have started yet)
    wait_count = 0
    while not progress_path.exists():
        if wait_count > 30:  # 15 seconds max wait
            yield {"event": "error", "data": "Progress file not created"}
            return
        await asyncio.sleep(0.5)
        wait_count += 1

        # Check if run is still active
        db = SessionLocal()
        run = db.query(Run).filter(Run.id == run_id).first()
        db.close()
        if run and run.status not in ("PENDING", "RUNNING"):
            yield {"event": "end", "data": run.status}
            return

    # Stream progress events
    while True:
        try:
            async with aiofiles.open(progress_path) as f:
                await f.seek(last_position)
                content = await f.read()

                if content:
                    lines = content.strip().split("\n")
                    for line in lines:
                        if line:
                            try:
                                event = json.loads(line)
                                # Update database with latest progress
                                _update_run_progress(run_id, event)
                                yield {"event": "progress", "data": json.dumps(event)}
                            except json.JSONDecodeError:
                                logger.warning("Invalid JSON in progress file: %s", line)

                    last_position = await f.tell()

        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error("Error reading progress file: %s", e)

        # Check if run is still active
        db = SessionLocal()
        run = db.query(Run).filter(Run.id == run_id).first()
        db.close()

        if run and run.status not in ("PENDING", "RUNNING"):
            yield {"event": "end", "data": run.status}
            break

        await asyncio.sleep(0.5)


def _update_run_progress(run_id: str, event: dict) -> None:
    """
    Update the run's progress fields in the database.

    :param run_id: The run identifier.
    :param event: Progress event dictionary.
    """
    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            if "erlang" in event:
                run.current_erlang = event["erlang"]
            if "iteration" in event:
                run.current_iteration = event["iteration"]
            if "total_iterations" in event:
                run.total_iterations = event["total_iterations"]
            if "total_erlangs" in event:
                run.total_erlangs = event["total_erlangs"]
            db.commit()
    except Exception as e:
        logger.error("Error updating run progress: %s", e)
        db.rollback()
    finally:
        db.close()


def parse_progress_file(progress_path: Path) -> list[dict]:
    """
    Parse all events from a progress.jsonl file.

    :param progress_path: Path to the progress.jsonl file.
    :returns: List of progress event dictionaries.
    """
    events = []
    if not progress_path.exists():
        return events

    try:
        with open(progress_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in progress file: %s", line)
    except Exception as e:
        logger.error("Error parsing progress file: %s", e)

    return events
