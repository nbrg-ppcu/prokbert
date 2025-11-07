import time
import pathlib
import argparse

import torch
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModel



dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'neuralbioinfo/prokbert-mini-long'
hidden_size = 384
batch_size = 896 # based on MFU calculations on A100 80GB


model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.to(device).to(dtype=dtype)
print("Using device:", device)


schema = pa.schema({
    "id": pa.int32(),
    "genom": pa.string(),
    "embedding": pa.list_(pa.float16(), list_size=hidden_size),
})

def to_table(ids, genoms, embeddings):
    # ids: torch.Tensor [B]
    # genoms: list[str] length B
    # embeddings: torch.Tensor [B, H]

    ids_np = ids.cpu().to(torch.int32).numpy()
    emb_np = embeddings.to(dtype=torch.float32).cpu().numpy()

    row = {
        "id": ids_np,                     # 1-D int32
        "genom": genoms,                  # list[str]
        "embedding": emb_np.tolist(),     # list[list[float16]]
    }
    return pa.Table.from_pydict(row, schema=schema)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_path", type=str, required=True)
    args = parser.parse_args()

    tokenized_dataset_path = args.chunk_path
    print(f"Loading dataset chunk: {tokenized_dataset_path}")
    dataset_chunk_tokenized = load_from_disk(tokenized_dataset_path)
    print(dataset_chunk_tokenized)

    ds_embedding_path = args.chunk_path + "_embeddings.arrow"

    if pathlib.Path(ds_embedding_path).exists():
        print(f"Embedding file already exists: {ds_embedding_path}")
        print("Skipping this chunk.")
        return

    print("Creating embedding writer at:", ds_embedding_path)
    writer = ipc.RecordBatchFileWriter(str(ds_embedding_path), schema)

    cols = ["input_ids", "attention_mask"]
    dataset_chunk_tokenized.set_format(type="torch", columns=cols, output_all_columns=True)


    loader = DataLoader(dataset_chunk_tokenized, batch_size=batch_size, pin_memory=True)

    start_time = time.time()

    for batch in tqdm(loader, total=len(loader), desc="Processing genes"):

        with torch.inference_mode(), torch.autocast(device_type=str(device), dtype=dtype):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            batch_row = to_table(batch["id"], batch["genom"], outputs.pooler_output)
            writer.write_table(batch_row)

    writer.close()

    # check that embeddings were written correctly, and dataset loads
    with pa.memory_map(str(ds_embedding_path), "r") as f:
        table = ipc.open_file(f).read_all()

    print(f"Done embedding processing: {args.chunk_path} in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
