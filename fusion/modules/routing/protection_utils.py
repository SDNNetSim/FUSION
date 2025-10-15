"""
Utility functions for 1+1 protection routing.

This module provides helper functions for spectrum reservation on dual paths
and failure handling for protection mechanisms.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def find_available_slots_on_path(
    network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
    path: list[int],
    slots_needed: int,
    band: str = "C",
    core: int = 0,
) -> set[int]:
    """
    Find available contiguous slots on a given path.

    :param network_spectrum_dict: Network spectrum state
    :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
    :param path: Path as list of node IDs
    :type path: list[int]
    :param slots_needed: Number of contiguous slots needed
    :type slots_needed: int
    :param band: Spectrum band (default: "C")
    :type band: str
    :param core: Core number (default: 0)
    :type core: int
    :return: Set of starting slot indices where slots are available
    :rtype: set[int]
    """
    if len(path) < 2:
        return set()

    # Get spectrum state for first link
    first_link = (path[0], path[1])
    if first_link not in network_spectrum_dict:
        logger.warning(f"Link {first_link} not in spectrum dict")
        return set()

    link_spectrum = network_spectrum_dict[first_link]

    # Handle different spectrum dict structures
    if "cores_matrix" in link_spectrum:
        # Multi-core structure
        if band not in link_spectrum["cores_matrix"]:
            logger.warning(f"Band {band} not in cores_matrix")
            return set()
        if core not in link_spectrum["cores_matrix"][band]:
            logger.warning(f"Core {core} not in band {band}")
            return set()
        spectrum_array = link_spectrum["cores_matrix"][band][core]
    elif "slots" in link_spectrum:
        # Simple structure
        spectrum_array = link_spectrum["slots"]
    else:
        logger.warning(f"Unknown spectrum dict structure for link {first_link}")
        return set()

    # Find available starting positions on first link
    available_starts = set()
    for start in range(len(spectrum_array) - slots_needed + 1):
        # Check if slots are free (value 0)
        if all(spectrum_array[start + i] == 0 for i in range(slots_needed)):
            available_starts.add(start)

    # Check availability on remaining links
    for i in range(1, len(path) - 1):
        link = (path[i], path[i + 1])
        if link not in network_spectrum_dict:
            logger.warning(f"Link {link} not in spectrum dict")
            return set()

        link_spectrum = network_spectrum_dict[link]

        if "cores_matrix" in link_spectrum:
            spectrum_array = link_spectrum["cores_matrix"][band][core]
        elif "slots" in link_spectrum:
            spectrum_array = link_spectrum["slots"]
        else:
            return set()

        # Remove starts that aren't available on this link
        available_starts = {
            start
            for start in available_starts
            if all(spectrum_array[start + j] == 0 for j in range(slots_needed))
        }

        if not available_starts:
            return set()

    return available_starts


def allocate_spectrum_on_path(
    network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
    path: list[int],
    start_slot: int,
    end_slot: int,
    request_id: int,
    band: str = "C",
    core: int = 0,
) -> None:
    """
    Allocate spectrum on a path for a request.

    :param network_spectrum_dict: Network spectrum state
    :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
    :param path: Path as list of node IDs
    :type path: list[int]
    :param start_slot: Starting slot index
    :type start_slot: int
    :param end_slot: Ending slot index (exclusive)
    :type end_slot: int
    :param request_id: Request identifier
    :type request_id: int
    :param band: Spectrum band (default: "C")
    :type band: str
    :param core: Core number (default: 0)
    :type core: int
    """
    for i in range(len(path) - 1):
        link = (path[i], path[i + 1])
        if link not in network_spectrum_dict:
            logger.warning(f"Link {link} not in spectrum dict")
            continue

        link_spectrum = network_spectrum_dict[link]

        if "cores_matrix" in link_spectrum:
            spectrum_array = link_spectrum["cores_matrix"][band][core]
        elif "slots" in link_spectrum:
            spectrum_array = link_spectrum["slots"]
        else:
            logger.warning(f"Unknown spectrum dict structure for link {link}")
            continue

        # Allocate slots
        for slot_idx in range(start_slot, end_slot):
            if slot_idx < len(spectrum_array):
                spectrum_array[slot_idx] = request_id


def reserve_spectrum_dual_path(
    network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
    primary_path: list[int],
    backup_path: list[int],
    slots_needed: int,
    request_id: int,
    band: str = "C",
    core: int = 0,
) -> tuple[int, int] | None:
    """
    Reserve spectrum on both primary and backup paths.

    Finds contiguous slots available on BOTH paths and reserves
    them simultaneously.

    :param network_spectrum_dict: Network spectrum state
    :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
    :param primary_path: Primary path
    :type primary_path: list[int]
    :param backup_path: Backup path
    :type backup_path: list[int]
    :param slots_needed: Number of contiguous slots
    :type slots_needed: int
    :param request_id: Request identifier
    :type request_id: int
    :param band: Spectrum band (default: "C")
    :type band: str
    :param core: Core number (default: 0)
    :type core: int
    :return: (start_slot, end_slot) or None if not available
    :rtype: tuple[int, int] | None

    Example:
        >>> result = reserve_spectrum_dual_path(
        ...     spectrum_dict,
        ...     primary=[0, 1, 2],
        ...     backup=[0, 3, 2],
        ...     slots_needed=4,
        ...     request_id=42
        ... )
        >>> print(result)
        (10, 14)  # Slots 10-13 reserved on both paths
    """
    # Find available slots on primary
    primary_slots = find_available_slots_on_path(
        network_spectrum_dict, primary_path, slots_needed, band, core
    )

    if not primary_slots:
        logger.debug("No available slots on primary path")
        return None

    # Find available slots on backup
    backup_slots = find_available_slots_on_path(
        network_spectrum_dict, backup_path, slots_needed, band, core
    )

    if not backup_slots:
        logger.debug("No available slots on backup path")
        return None

    # Find common available slot ranges
    common_slots = primary_slots.intersection(backup_slots)

    if not common_slots:
        logger.debug(
            f"No common slots: primary has {len(primary_slots)}, "
            f"backup has {len(backup_slots)}"
        )
        return None

    # Select first available common range
    start_slot = min(common_slots)
    end_slot = start_slot + slots_needed

    logger.debug(
        f"Dual-path reservation: slots {start_slot}-{end_slot - 1} "
        f"for request {request_id}"
    )

    # Reserve on both paths
    allocate_spectrum_on_path(
        network_spectrum_dict,
        primary_path,
        start_slot,
        end_slot,
        request_id,
        band,
        core,
    )

    allocate_spectrum_on_path(
        network_spectrum_dict, backup_path, start_slot, end_slot, request_id, band, core
    )

    return start_slot, end_slot


def release_spectrum_on_path(
    network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
    path: list[int],
    start_slot: int,
    end_slot: int,
    band: str = "C",
    core: int = 0,
) -> None:
    """
    Release spectrum on a path.

    :param network_spectrum_dict: Network spectrum state
    :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
    :param path: Path as list of node IDs
    :type path: list[int]
    :param start_slot: Starting slot index
    :type start_slot: int
    :param end_slot: Ending slot index (exclusive)
    :type end_slot: int
    :param band: Spectrum band (default: "C")
    :type band: str
    :param core: Core number (default: 0)
    :type core: int
    """
    for i in range(len(path) - 1):
        link = (path[i], path[i + 1])
        if link not in network_spectrum_dict:
            continue

        link_spectrum = network_spectrum_dict[link]

        if "cores_matrix" in link_spectrum:
            spectrum_array = link_spectrum["cores_matrix"][band][core]
        elif "slots" in link_spectrum:
            spectrum_array = link_spectrum["slots"]
        else:
            continue

        # Release slots (set to 0)
        for slot_idx in range(start_slot, end_slot):
            if slot_idx < len(spectrum_array):
                spectrum_array[slot_idx] = 0


def release_spectrum_dual_path(
    network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
    primary_path: list[int],
    backup_path: list[int],
    start_slot: int,
    end_slot: int,
    band: str = "C",
    core: int = 0,
) -> None:
    """
    Release spectrum on both primary and backup paths.

    :param network_spectrum_dict: Network spectrum state
    :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
    :param primary_path: Primary path
    :type primary_path: list[int]
    :param backup_path: Backup path
    :type backup_path: list[int]
    :param start_slot: Starting slot index
    :type start_slot: int
    :param end_slot: Ending slot index (exclusive)
    :type end_slot: int
    :param band: Spectrum band (default: "C")
    :type band: str
    :param core: Core number (default: 0)
    :type core: int
    """
    release_spectrum_on_path(
        network_spectrum_dict, primary_path, start_slot, end_slot, band, core
    )

    release_spectrum_on_path(
        network_spectrum_dict, backup_path, start_slot, end_slot, band, core
    )
