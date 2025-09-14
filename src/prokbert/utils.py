from typing import Any, Optional

import time
import torch
from thop import profile

# theoretical max flops per second provided by the GPU manufacturer
FLOPS_PER_SECOND = {
    # https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899
    "H100": {
        torch.float32: 51.22e12,  # 51.22 TFLOPs for FP32 on NVIDIA H100
        torch.float16: 204.9e12,  # 204.9 TFLOPs for FP16 on NVIDIA H100
        torch.bfloat16: 204.9e12
    },
    # https://www.techpowerup.com/gpu-specs/l4.c4091
    "L4": {
        torch.float32: 30.29e12,  # 30.29 TFLOPs for FP32 on NVIDIA L4
        torch.float16: 30.29e12,  # 30.29 TFLOPs for FP16 on NVIDIA L4
        torch.bfloat16: 30.29e12
    },
    # https://www.techpowerup.com/gpu-specs/tesla-t4.c3316
    "T4": {
        torch.float32: 8.1e12,  # 8.1 TFLOPs for FP32 on NVIDIA T4
        torch.float16: 65.13e12,  # 65.13 TFLOPs for FP16 on NVIDIA T4
        torch.bfloat16: 65.13e12
    },
    # https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623
    "A100": {
        torch.float32: 19.49e12,  # 19.49 TFLOPs for FP32 on NVIDIA A100
        torch.float16: 77.97e12,  # 77.97 TFLOPs for FP16 on NVIDIA A100
        torch.bfloat16: 77.97e12
    },
    # https://www.techpowerup.com/gpu-specs/h200-nvl.c4254
    "H200": {
        torch.float32: 60.32e12,  # 60.32 TFLOPs for FP32 on NVIDIA H200
        torch.float16: 241.3e12,  # 241.3 TFLOPs for FP16 on NVIDIA H200
        torch.bfloat16: 241.3e12
    }
}

def compute_flops(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        device: torch.device
    ) -> tuple[Any, Any]:
    model = model.bfloat16()
    model.to(device)
    # MACS = multiply-accumulate operations
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2 * macs
    print(f"Model has {flops:.1e} FLOPS and {params / 1e6:.2f} M parameters.")
    return flops, params


def compute_model_params(model: torch.nn.Module) -> int:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params / 1e6:.2f} M")
    return params


def get_gpu_model(flops_per_second_dict):
    device_name = torch.cuda.get_device_name(0)
    for model in flops_per_second_dict.keys():
        if model in device_name:
            return model
    raise ValueError(
        f"GPU model '{device_name}' not found in "
        "the provided FLOPS per second dictionary:"
        f" {flops_per_second_dict.keys()}."
        )


def compute_mfu(
        model: torch.nn.Module,
        context_length: int,
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
    """
    Model FLOP Utilization (MFU)
    """
    if (vocab_size is None) == (embedding_dim is None):
        raise ValueError("You must specify exactly one of vocab_size or embedding_dim.")

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu_model = get_gpu_model(FLOPS_PER_SECOND)

    min_batch_size = 1
    max_batch_size = None
    max_possible_batch_size = 4096

    while min_batch_size <= max_possible_batch_size:
        batch_size = (min_batch_size + max_possible_batch_size) // 2
        try:
            if embedding_dim is not None:
                input_tensor = torch.randn(
                    batch_size,
                    context_length,
                    embedding_dim,
                    dtype=torch.float32,
                    device=device
                )
            elif vocab_size is not None:
                input_tensor = torch.randint(
                    0, vocab_size,
                    (batch_size, context_length),
                    device=device
                )

            model = model.to(dtype).to(device)
            model.train()

            # start timing
            torch.cuda.synchronize()
            start_time = time.time()

            # forward & backward pass
            output = model(input_tensor)
            loss = output[0].loss.backward() # dummy loss, sum over last hidden state

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