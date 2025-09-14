import time
import torch
from thop import profile

from prokbert.utils import get_gpu_model, FLOPS_PER_SECOND
from prokbert.model.genome_network.modeling_genome_network import GenomeNetwork
from prokbert.model.genome_network.configuration_genome_network import GenomeNetworkConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = GenomeNetworkConfig()


context_length = 1024
embedding_dim = 384
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_model = get_gpu_model(FLOPS_PER_SECOND)
print(f"GPU Model: {gpu_model} with dtype {dtype}.")

min_batch_size = 1
max_batch_size = None
max_possible_batch_size = 4096


while min_batch_size <= max_possible_batch_size:
    batch_size = (min_batch_size + max_possible_batch_size) // 2
    try:
        input_tensor = torch.randn(
            batch_size,
            context_length,
            embedding_dim,
            dtype=dtype,
            device=device
        )
        model = GenomeNetwork(config)
        model = model.to(dtype).to(device)
        model.train()

        # start timing
        torch.cuda.synchronize()
        start_time = time.time()

        # forward & backward pass
        output = model(input_tensor)
        loss = output[0].sum().backward() # dummy loss, sum over last hidden state

        # end timing
        torch.cuda.synchronize()
        end_time = time.time()

        total_time_seconds = end_time - start_time

        # calculate FLOPs for forward pass
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_forward = 2 * macs  # Assuming one MAC equals two FLOPs

        # estimate FLOPs for backward pass (typically 2x forward FLOPs)
        flops_backward = 2 * flops_forward

        # total FLOPs for forward + backward passes
        total_flops = flops_forward + flops_backward  # Or total_flops = flops_forward * 3

        data_type = next(model.parameters()).dtype
        max_flops_per_second = FLOPS_PER_SECOND[gpu_model].get(data_type, 0)

        # compute tokens per second
        tokens_processed = batch_size * context_length
        tokens_per_second = tokens_processed / total_time_seconds

        # compute FLOPs per token
        flops_per_token = total_flops / tokens_processed

        # compute theoretical max tokens per second
        if flops_per_token > 0:
            theoretical_max_tokens_per_second = max_flops_per_second / flops_per_token
        else:
            theoretical_max_tokens_per_second = 0  # avoid division by zero

        # compute MFU
        if theoretical_max_tokens_per_second > 0:
            mfu = tokens_per_second / theoretical_max_tokens_per_second
        else:
            mfu = 0  # avoid division by zero

        print(f"  Batch size {batch_size}: Tokens/sec: {tokens_per_second:.2f}, MFU: {mfu:.4f}")

        # if successful, try a larger batch size
        min_batch_size = batch_size + 1
        max_batch_size = batch_size

        # clean up
        del model, input_tensor, output, loss
        torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # try smaller batch size
            max_possible_batch_size = batch_size - 1

            # clean up
            try:
                del model, input_tensor
                torch.cuda.empty_cache()
            except NameError:
                pass
        else:
            raise e
