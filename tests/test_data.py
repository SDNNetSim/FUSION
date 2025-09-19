"""
Shared test data for unity module tests.

Contains common test fixtures and data structures used across multiple
test files to avoid code duplication.
"""

# Common SLURM resource configuration for tests
TEST_RESOURCES = {
    "partition": "gpu",
    "time": "24:00:00",
    "mem": "32G",
    "cpus": "8",
    "gpus": "1",
    "nodes": "1"
}
