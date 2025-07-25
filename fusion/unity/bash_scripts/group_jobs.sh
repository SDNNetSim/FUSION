#!/bin/bash

# This script calculates and displays resource usage (CPUs, memory, nodes) for a list of users
# in a specific group over a specified period and for currently running jobs on the SLURM cluster.

# Usage: ./resource_usage.sh <start_date>
# Example: ./resource_usage.sh '7 days ago'

# Check if at least one argument is provided (the start date)
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <start_date>"
  echo "Example: $0 '7 days ago'"
  exit 1
fi

# Parse the start date and format it as YYYY-MM-DD
START_DATE=$(date -d "$1" "+%Y-%m-%d")

# Define the list of users to check resource usage for
GROUP_USERS=("ryan_mccann_student_uml_edu" "arash_rezaee_student_uml_edu" "shamsunnahar_edib_student_uml_edu" "kenneth_watts_student_uml_edu")

# Function to print resource usage for a given user
print_user_resources() {
  local user=$1
  local is_current=$2
  local total_cpus=0
  local total_mem_mb=0
  local total_nodes=0
  local job_info=""

  # Get job information based on whether we are checking current jobs or past jobs since START_DATE
  if [ "$is_current" = true ]; then
    job_info=$(squeue -u $user --noheader --format="%i %C %m %D")
  else
    job_info=$(sacct -u $user --noheader -o JobID,AllocCPUS,ReqMem,AllocNodes -S "$START_DATE")
  fi

  # Process each line of job information
  while IFS= read -r line; do
    local jobid=$(echo $line | awk '{print $1}')
    local cpus=$(echo $line | awk '{print $2}')
    local mem_spec=$(echo $line | awk '{print $3}')
    local nodes=$(echo $line | awk '{print $4}')

    local mem_value=${mem_spec%[cCnN]*}
    local mem_type=${mem_spec: -1}

    # Convert memory specification to MB
    if [[ $mem_value =~ "G" ]]; then
      mem_value=$(echo $mem_value | tr -d 'G')
      mem_value=$((mem_value * 1024))
    else
      mem_value=$(echo $mem_value | tr -d 'M')
    fi

    # Calculate total memory based on memory type
    if [ "$mem_type" == "c" ] || [ "$mem_type" == "C" ]; then
      total_mem_mb=$((total_mem_mb + mem_value * cpus))
    else
      total_mem_mb=$((total_mem_mb + mem_value))
    fi

    total_cpus=$((total_cpus + cpus))
    total_nodes=$((total_nodes + nodes))
  done <<<"$job_info"

  # Convert total memory to GB
  local total_mem_gb=$(echo "scale=2; $total_mem_mb / 1024" | bc)

  # Print resource usage for the user
  echo "User $user is using $total_cpus CPUs, $total_mem_gb GB (or $total_mem_mb MB) of memory, and $total_nodes nodes."
}

# Print resource usage for past jobs since START_DATE
echo "Resource usage from $START_DATE to now:"
for group_user in "${GROUP_USERS[@]}"; do
  print_user_resources $group_user false
done

echo ""
# Print resource usage for currently running jobs
echo "Resource usage for currently running jobs:"
for group_user in "${GROUP_USERS[@]}"; do
  print_user_resources $group_user true
done
