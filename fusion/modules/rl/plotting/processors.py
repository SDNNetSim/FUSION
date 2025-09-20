# Standard library imports
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

# Third-party imports
import numpy as np
from scipy.stats import ttest_ind


def _mean_last(values: list[float | int], k: int = 5) -> float:
    """Calculate mean of last k values in the list."""
    if not values:
        return 0.0
    subset = values[-k:] if len(values) >= k else values
    return float(np.mean(subset))


def _collect_baseline_data(merged_data: dict, baseline_algorithms: list[str]) -> dict:
    """Extract baseline algorithm data for statistical comparison."""
    baseline_values = defaultdict(dict)
    for baseline_algo in baseline_algorithms:
        for traffic_volume, values in merged_data[baseline_algo].items():
            baseline_values[baseline_algo][traffic_volume] = np.array(
                values, dtype=float
            )
    return baseline_values


def _calculate_statistics(values: np.ndarray) -> dict:
    """Calculate mean, standard deviation, and confidence interval for values."""
    mean_value = np.mean(values)
    std_value = np.std(values, ddof=1) if len(values) > 1 else 0.0
    ci_value = 1.96 * (std_value / np.sqrt(len(values))) if len(values) > 1 else 0.0

    return {
        "mean": float(mean_value),
        "std": float(std_value),
        "ci": float(ci_value),
    }


def _compute_effect_sizes(
    algorithm_values: np.ndarray, baseline_values: np.ndarray, baseline_name: str
) -> dict:
    """Compute statistical comparisons vs baseline algorithm."""
    if len(baseline_values) <= 1 or len(algorithm_values) <= 1:
        return {}

    _, p_value = ttest_ind(algorithm_values, baseline_values, equal_var=False)
    pooled_std = np.sqrt(
        (np.var(algorithm_values, ddof=1) + np.var(baseline_values, ddof=1)) / 2
    )
    cohens_d = (
        (np.mean(algorithm_values) - np.mean(baseline_values)) / pooled_std
        if pooled_std
        else 0.0
    )
    mean_difference = np.mean(algorithm_values) - np.mean(baseline_values)

    return {
        f"vs_{baseline_name}": {
            "p": float(p_value),
            "d": float(cohens_d),
            "mean_diff": float(mean_difference),
            "significant": p_value < 0.05,
        }
    }


def process_blocking(raw_runs: dict[str, Any], runid_to_algo: dict[str, str]) -> dict:
    """
    Process blocking probability results into mean, std, CI and optionally effect sizes.
    """
    merged_data = defaultdict(lambda: defaultdict(list))
    baseline_algorithms = [
        "k_shortest_path_1",
        "k_shortest_path_4",
        "cong_aware",
        "k_shortest_path_inf",
    ]

    # Collect data from all runs
    for run_id, data in raw_runs.items():
        algorithm = runid_to_algo.get(run_id, "unknown")
        for traffic_volume, info_vector in data.items():
            if isinstance(info_vector, dict):
                last_key = next(reversed(info_vector["iter_stats"]))
                last_entry = info_vector["iter_stats"][last_key]
                if algorithm in baseline_algorithms:
                    blocking_list = last_entry["sim_block_list"]
                    merged_data[algorithm][str(traffic_volume)] = blocking_list
                elif (
                    info_vector.get("blocking_mean") is None
                    and "iter_stats" in info_vector
                ):
                    merged_data[algorithm][str(traffic_volume)].append(
                        _mean_last(last_entry["sim_block_list"])
                    )
                else:
                    raise NotImplementedError
            elif isinstance(info_vector, (float, int)):
                merged_data[algorithm][str(traffic_volume)].append(float(info_vector))

    # Extract baseline data for comparisons
    baseline_values = _collect_baseline_data(merged_data, baseline_algorithms)

    # Process each algorithm
    processed = {}
    for algorithm, traffic_volume_dict in merged_data.items():
        processed[algorithm] = {}
        for traffic_volume, values in traffic_volume_dict.items():
            values_array = np.array(values, dtype=float)
            statistics_block = _calculate_statistics(values_array)

            # Add comparisons to baselines for non-baseline algorithms
            if algorithm not in baseline_algorithms:
                for baseline_algo in baseline_algorithms:
                    baseline_vals = np.array(
                        baseline_values.get(baseline_algo, {}).get(traffic_volume, []),
                        dtype=float,
                    )
                    effect_size_stats = _compute_effect_sizes(
                        values_array, baseline_vals, baseline_algo
                    )
                    statistics_block.update(effect_size_stats)

            processed[algorithm][traffic_volume] = statistics_block

            # Debug logging
            print(
                f"[SEED-DBG] {algorithm} Erlang={traffic_volume} seeds={len(values_array)} "
                f"mean={statistics_block['mean']:.4g} ±std={statistics_block['std']:.4g} ±CI={statistics_block['ci']:.4g}"
            )

    return processed


def _add(
    collector: dict[str, dict[str, list[float]]],
    algorithm: str,
    traffic_volume: str,
    value: float | list[float] | np.ndarray,
) -> None:
    """Append one or many numeric values to collector[algorithm][traffic_volume].

    :param collector: Nested dictionary to store values
    :type collector: Dict[str, Dict[str, List[float]]]
    :param algorithm: Algorithm name
    :type algorithm: str
    :param traffic_volume: Traffic volume identifier
    :type traffic_volume: str
    :param value: Numeric value(s) to append
    :type value: Union[float, List[float], np.ndarray]
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        collector[algorithm][traffic_volume].extend(map(float, value))
    else:
        collector[algorithm][traffic_volume].append(float(value))


def process_memory_usage(
    raw_runs: dict[str, Any], runid_to_algo: dict[str, str]
) -> dict:
    """
    Aggregate memory usage (MB).

    * DRL runs → { 'overall': float }   from memory_usage.npy
    * Legacy runs → float / list / ndarray keyed by traffic volume
    """
    merged_data = defaultdict(lambda: defaultdict(list))

    for run_id, data in raw_runs.items():
        algorithm = runid_to_algo.get(run_id, "unknown")
        traffic_volume = next(iter(data))
        merged_data[algorithm][traffic_volume] = {"overall": data.get("overall", -1.0)}

    return merged_data


def _stamp_to_dt(timestamp_string: str) -> datetime:
    """
    Convert timestamp string like '0429_21_14_39_491949' to a datetime object.

    :param timestamp_string: Formatted timestamp string
    :type timestamp_string: str
    :return: Corresponding datetime object
    :rtype: datetime
    :raises ValueError: If timestamp format is invalid
    """
    try:
        month_day, hour, minute, second, microseconds_str = timestamp_string.split("_")
        milliseconds = int(microseconds_str[:3])
        microseconds = int(microseconds_str[3:])
        return datetime(
            year=datetime.now().year,
            month=int(month_day[:2]),
            day=int(month_day[2:]),
            hour=int(hour),
            minute=int(minute),
            second=int(second),
            microsecond=milliseconds * 1000 + microseconds,
        )
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid timestamp format: {timestamp_string}") from exc


def process_sim_times(
    raw_runs: dict[str, Any],
    runid_to_algo: dict[str, str],
    context: dict | None = None,
) -> dict:
    """
    Compute wall-clock durations or fallback to reported simulation times.
    """
    start_timestamps = context.get("start_stamps") if context else None
    merged_data = defaultdict(lambda: defaultdict(list))

    for run_id, data in raw_runs.items():
        algorithm = runid_to_algo.get(run_id, "unknown")

        if isinstance(start_timestamps, dict) and run_id in start_timestamps:  # pylint: disable=unsupported-membership-test
            start_time = _stamp_to_dt(start_timestamps[run_id])

            for traffic_volume, info in data.items():
                if not isinstance(info, dict):
                    continue
                end_timestamp_raw = info.get("sim_end_time")
                if not end_timestamp_raw:
                    continue

                end_time = _stamp_to_dt(end_timestamp_raw)
                if end_time < start_time:
                    end_time += timedelta(days=1)

                merged_data[algorithm][str(traffic_volume)].append(
                    (end_time - start_time).total_seconds()
                )
        else:
            for traffic_volume, seconds in data.items():
                merged_data[algorithm][str(traffic_volume)].append(float(seconds))

    return {
        algorithm: {
            traffic_volume: float(np.mean(values))
            for traffic_volume, values in traffic_volume_dict.items()
        }
        for algorithm, traffic_volume_dict in merged_data.items()
    }
