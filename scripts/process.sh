#!/bin/bash
set -e

DATASET_PATH=""
PYTHON_SCRIPT=""
MAX_JOBS=8
SECONDS=0

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

# checks how many background jobs are running
sem() {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        sleep 1
    done
}

echo "Starting batch processing..."
echo "Using chunks in: $DATASET_PATH"
echo "Max parallel jobs: $MAX_JOBS"
echo


for d in "$DATASET_PATH"/dataset_*; do
    [ -d "$d" ] || continue # skip if not a directory
    sem  # wait until a slot is free
    echo "Starting: $d"
    python "$PYTHON_SCRIPT" --chunk_path "$d" &
done

wait # for all background jobs to finish

TOTAL_TIME=$SECONDS
hours=$(( TOTAL_TIME / 3600 ))
minutes=$(( (TOTAL_TIME % 3600) / 60 ))
seconds=$(( TOTAL_TIME % 60 ))

echo "All chunks processed."
echo "Total time: ${hours}h ${minutes}m ${seconds}s"