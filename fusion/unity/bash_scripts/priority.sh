#!/bin/bash

# This script checks the job priority of a specified user in a specified SLURM partition.
# It determines the user's highest priority for pending jobs and compares it with other jobs in the partition.

# Usage: ./check_priority.sh <username> <partition>
# Example: ./check_priority.sh username partition

# Check if at least two arguments are provided (username and partition)
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <username> <partition>"
  exit 1
fi

# Assign arguments to variables for better readability
USER_NAME=$1
PARTITION=$2

# Get the highest priority of the user's pending jobs in the specified partition
USER_PRIORITY=$(squeue -p $PARTITION -t PD --noheader --format="%u %Q" | grep "^$USER_NAME " | awk '{print $2}' | sort -nr | head -n 1)

# If the user has pending jobs, count how many jobs have a higher priority than the user's highest priority job
if [[ ! -z $USER_PRIORITY ]]; then
  HIGHER_PRIORITY_JOBS=$(squeue -p $PARTITION -t PD --noheader --format="%Q" | awk -v up="$USER_PRIORITY" '$1 > up' | wc -l)
fi

# Get the highest priority in the queue for the specified partition
HIGHEST_PRIORITY=$(squeue -p $PARTITION -t PD --noheader --format="%Q" | sort -nr | head -n 1)

# Check if the user has any pending jobs and print the appropriate message
if [[ -z $USER_PRIORITY ]]; then
  echo "User: $USER_NAME has no pending jobs in the '$PARTITION' partition."
else
  echo "User: $USER_NAME has priority $USER_PRIORITY in the '$PARTITION' partition, and the highest priority in the queue is currently $HIGHEST_PRIORITY."
  echo "$HIGHER_PRIORITY_JOBS jobs have a higher priority than $USER_NAME's highest priority job in the '$PARTITION' partition."
fi
