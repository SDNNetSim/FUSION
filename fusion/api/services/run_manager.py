"""
Run manager service for simulation lifecycle.

Handles:
- Creating and starting simulation runs
- Process management and cancellation
- Log streaming via SSE
- Recovery of orphaned runs on startup
"""

import asyncio
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

import aiofiles
from sqlalchemy.orm import Session

from ..config import settings
from ..db.models import Run
from ..schemas.run import RunCreate

logger = logging.getLogger(__name__)


class RunManager:
    """Manages simulation run lifecycle."""

    def __init__(self, db: Session) -> None:
        """
        Initialize the run manager.

        :param db: SQLAlchemy database session.
        """
        self.db = db

    def create_run(self, data: RunCreate) -> Run:
        """
        Create a new run and start the simulation.

        :param data: Run creation request.
        :returns: The created Run object.
        :raises ValueError: If maximum concurrent runs reached.
        """
        # Check concurrency limit
        active = self.db.query(Run).filter(Run.status == "RUNNING").count()
        if active >= settings.max_concurrent_runs:
            raise ValueError(
                f"Maximum concurrent runs ({settings.max_concurrent_runs}) reached"
            )

        # Generate ID and paths
        run_id = uuid.uuid4().hex[:12]
        run_dir = settings.runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "output").mkdir()

        # Write config
        config_path = run_dir / "config.ini"
        self._write_config(data, config_path)

        # Create database record
        run = Run(
            id=run_id,
            name=data.name or f"Run {run_id[:6]}",
            status="PENDING",
            template=data.template,
            config_json=json.dumps(data.config),
        )
        self.db.add(run)
        self.db.commit()

        # Start simulation
        self._start_process(run, config_path)

        logger.info("Created run %s with template %s", run_id, data.template)
        return run

    def get_run(self, run_id: str) -> Run | None:
        """
        Get a run by ID.

        :param run_id: The run identifier.
        :returns: The Run object or None if not found.
        """
        return self.db.query(Run).filter(Run.id == run_id).first()

    def cancel_or_delete(self, run_id: str) -> Run | None:
        """
        Cancel a running job or delete a completed job.

        :param run_id: The run identifier.
        :returns: The Run object or None if not found.
        """
        run = self.db.query(Run).filter(Run.id == run_id).first()
        if not run:
            return None

        if run.status == "RUNNING":
            self._kill_process(run)
            logger.info("Cancelled running job %s", run_id)

        # Delete artifacts
        run_dir = settings.runs_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)

        self.db.delete(run)
        self.db.commit()

        logger.info("Deleted run %s", run_id)
        return run

    def _start_process(self, run: Run, config_path: Path) -> None:
        """
        Start the simulation subprocess.

        :param run: The Run database object.
        :param config_path: Path to the config file.
        """
        run_dir = settings.runs_dir / run.id
        log_path = run_dir / "logs" / "sim.log"
        progress_path = run_dir / "progress.jsonl"

        with open(log_path, "w") as log_file:
            # Build command
            cmd = [
                sys.executable,
                "-m",
                "fusion.cli.run_sim",
                "run_sim",
                "--config_path",
                str(config_path),
                "--run_id",
                run.id,
                "--progress_file",
                str(progress_path),
            ]

            # Platform-specific process creation
            if platform.system() == "Windows":
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
                run.pgid = None  # Windows doesn't use pgid
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # New process group
                )
                run.pgid = os.getpgid(process.pid)

        run.status = "RUNNING"
        run.pid = process.pid
        run.started_at = datetime.utcnow()
        self.db.commit()

        logger.info("Started process %d for run %s", process.pid, run.id)

    def _kill_process(self, run: Run) -> None:
        """
        Kill the entire process group.

        :param run: The Run database object.
        """
        if not run.pid:
            return

        try:
            if platform.system() == "Windows":
                # Windows: use taskkill to kill process tree
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(run.pid)],
                    capture_output=True,
                )
            else:
                # POSIX: kill process group
                if run.pgid:
                    os.killpg(run.pgid, signal.SIGTERM)
                    time.sleep(2)
                    try:
                        os.killpg(run.pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except ProcessLookupError:
            pass
        except OSError as e:
            logger.warning("Error killing process %d: %s", run.pid, e)

        run.status = "CANCELLED"
        run.completed_at = datetime.utcnow()
        self.db.commit()

    def _write_config(self, data: RunCreate, path: Path) -> None:
        """
        Write configuration to INI file.

        :param data: Run creation request.
        :param path: Path to write the config file.
        """
        template_path = Path(settings.templates_dir) / f"{data.template}.ini"
        if not template_path.exists():
            template_path = Path(settings.templates_dir) / "default.ini"

        if template_path.exists():
            shutil.copy(template_path, path)
        else:
            # Create minimal config if no template exists
            path.write_text("[general_settings]\n")

        # TODO: Merge template with overrides from data.config


def recover_orphaned_runs() -> None:
    """
    Mark stale RUNNING jobs as FAILED on startup.

    Called during application startup to handle runs that were
    active when the server was stopped.
    """
    from ..db.database import SessionLocal

    db = SessionLocal()
    try:
        running = db.query(Run).filter(Run.status == "RUNNING").all()
        for run in running:
            if not _is_process_alive(run.pid, run.pgid):
                run.status = "FAILED"
                run.error_message = "Server restarted while run was active"
                run.completed_at = datetime.utcnow()
                logger.warning("Marked orphaned run %s as FAILED", run.id)
        db.commit()
    finally:
        db.close()


def _is_process_alive(pid: int | None, pgid: int | None) -> bool:
    """
    Check if a process is still running.

    :param pid: Process ID.
    :param pgid: Process group ID (POSIX only).
    :returns: True if the process is alive.
    """
    if not pid:
        return False

    if platform.system() == "Windows":
        # Windows: check if process exists
        try:
            subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False
    else:
        # POSIX: check process group
        if not pgid:
            return False
        try:
            os.killpg(pgid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


async def stream_run_logs(run_id: str, from_start: bool) -> AsyncGenerator[dict, None]:
    """
    Stream log file content via SSE.

    :param run_id: The run identifier.
    :param from_start: Whether to send existing content first.
    :yields: SSE event dictionaries.
    """
    log_path = settings.runs_dir / run_id / "logs" / "sim.log"

    if not log_path.exists():
        yield {"event": "error", "data": "Log file not found"}
        return

    async with aiofiles.open(log_path, mode="r") as f:
        if from_start:
            content = await f.read()
            if content:
                yield {"event": "log", "data": content}
        else:
            await f.seek(0, 2)  # End of file

        while True:
            line = await f.readline()
            if line:
                yield {"event": "log", "data": line.rstrip()}
            else:
                # Check if run is still active
                from ..db.database import SessionLocal

                db = SessionLocal()
                run = db.query(Run).filter(Run.id == run_id).first()
                db.close()

                if run and run.status not in ("PENDING", "RUNNING"):
                    yield {"event": "end", "data": run.status}
                    break

                await asyncio.sleep(0.3)
