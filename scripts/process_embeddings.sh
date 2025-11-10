#!/bin/bash
set -e

DATASET_PATH=""
PYTHON_SCRIPT=""

# 3 GPUs, 2 processes per GPU => 6 total
GPU_IDS=(0 1 2)
MAX_JOBS=6

SECONDS=0

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

# sanity check: GPUs array not empty
NGPUS=${#GPU_IDS[@]}
if (( NGPUS == 0 )); then
  echo "Error: GPU_IDS array is empty"; exit 1
fi

# wait until NO background jobs remain, avoid OOM on GPUs
sem_all() {
    while (( $(jobs -rp | wc -l) > 0 )); do
        sleep 1
    done
}

echo "Starting batch processing..."
echo "Using chunks in: $DATASET_PATH"
echo "Max parallel jobs: $MAX_JOBS (2 per GPU across ${#GPU_IDS[@]} GPUs)"
echo

count=0
batch_size=6

for d in "$DATASET_PATH"/dataset_*_tokenized; do
    [ -d "$d" ] || continue # skip if not a directory

    # if batch is full -> wait for all to finish
    if (( count > 0 && count % batch_size == 0 )); then
        sem_all
    fi

    # round-robin GPU assignment: 0,1,2,0,1,2,...
    idx=$(( count % NGPUS ))
    gpu=${GPU_IDS[$idx]}

    (( count += 1 ))

    echo "Starting: $d on GPU $gpu"

    # restrict the process to a single GPU (visible as cuda:0 inside the script)
    CUDA_VISIBLE_DEVICES="$gpu" python3 "$PYTHON_SCRIPT" --chunk_path "$d" &

done

sem_all # wait for the last batch

TOTAL_TIME=$SECONDS
hours=$(( TOTAL_TIME / 3600 ))
minutes=$(( (TOTAL_TIME % 3600) / 60 ))
seconds=$(( TOTAL_TIME % 60 ))

echo "All chunks processed."
echo "Total time: ${hours}h ${minutes}m ${seconds}s"