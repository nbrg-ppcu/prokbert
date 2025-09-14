from typing import Any

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
