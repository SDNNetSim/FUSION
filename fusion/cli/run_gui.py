"""
CLI entry point for launching the FUSION GUI interface.

Starts the FastAPI server with uvicorn for the web-based GUI.

Usage:
    fusion gui [--host HOST] [--port PORT] [--reload]
    python -m fusion.cli.run_gui [--host HOST] [--port PORT] [--reload]
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Launch the FUSION GUI web interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Launch the FUSION GUI interface.

    Starts a uvicorn server hosting the FastAPI application.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        logger.error(
            "uvicorn is not installed. Install the GUI dependencies with: "
            "pip install fusion[gui]"
        )
        return 1

    logger.info("Starting FUSION GUI at http://%s:%d", args.host, args.port)

    try:
        uvicorn.run(
            "fusion.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down FUSION GUI...")
    except Exception as e:
        logger.exception("Failed to start FUSION GUI: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
