import time
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer


num_cores = 8

model_name = 'neuralbioinfo/prokbert-mini-long'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

cls_id = tokenizer.cls_token_id
sep_id = tokenizer.sep_token_id

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["sequence"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"  # ignored by dataset .map()
    )
    # find indices of SEP token in each row
    input_ids = tokenized['input_ids']
    _, sep_id_indices_cols = (input_ids == sep_id).nonzero(as_tuple=True)

    attention_masks = tokenized['attention_mask']
    for i, attention_mask in enumerate(attention_masks):
        # CLS token attention mask to 0
        attention_mask[0] = 0
        # SEP token attention mask to 0
        sep_index = sep_id_indices_cols[i].item()
        attention_mask[sep_index] = 0

    return tokenized


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_path", type=str, required=True)
    args = parser.parse_args()

    # print(f"Loading dataset chunk: {args.chunk_path}")
    dataset = load_from_disk(args.chunk_path)

    start_time = time.time()

    dataset = dataset.map(tokenize_function, batched=True, num_proc=num_cores)
    dataset = dataset.remove_columns(["sequence", "token_type_ids"])

    save_path = args.chunk_path + "_tokenized"
    print("Saving updated chunk in : ", save_path)
    dataset.save_to_disk(save_path)

    print(f"Done processing: {args.chunk_path} in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()