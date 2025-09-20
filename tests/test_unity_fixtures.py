"""
Shared test fixtures for Unity module tests.
"""

# Common resource configuration used across Unity tests
COMMON_RESOURCES = {
    "partition": "gpu",
    "time": "24:00:00",
    "mem": "32G",
    "cpus": "8",
    "gpus": "1",
    "nodes": "1",
}
