import os
import time
import math
import pathlib
from datasets import load_from_disk


DATASET_PATH = pathlib.Path("")
OUTPUT_DIR = pathlib.Path("")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_PARTS = 328 # ~500k seqs per part

dataset = load_from_disk(str(DATASET_PATH))
print(dataset)

total_rows = len(dataset)
print(f"Total rows: {total_rows} ({total_rows / 1e6:.2f} million)")

chunk_size = math.ceil(total_rows / NUM_PARTS)
print(f"Chunk size: {chunk_size}") # 499 723 rows per chunk

start_time = time.time()
for i in range(NUM_PARTS):
    start = i * chunk_size
    end = min(start + chunk_size, total_rows)

    if start >= total_rows:
        break

    folder_name = f"dataset_{start}_{end}"
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    ds_slice = dataset.select(range(start, end))
    ds_slice.save_to_disk(folder_path)

    print(f"Creating shard {i+1}/{NUM_PARTS}: rows {start} -> {end} (folder name: {folder_name})")

print(f"Done! All dataset parts saved in {time.time() - start_time:.2f} seconds.")
# Done! All dataset parts saved in 510.34 seconds.