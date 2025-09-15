# coding=utf-8
import warnings
import logging
from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import MegatronBertConfig, MegatronBertModel, MegatronBertForMaskedLM, MegatronBertPreTrainedModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

import math
from contextlib import nullcontext
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
)
from transformers.utils.import_utils import is_triton_available
#from .prokbert2_config import ProkBertConfig

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    RotaryEmbedding = object

logger = logging.get_logger(__name__)

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.ops.triton.rotary import apply_rotary


from typing import Literal
from transformers.configuration_utils import PretrainedConfig




VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# Define the mapping for pretrained vocabulary files
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "lca-mini-k6s1": "lca-base-dna6/vocab.txt",
        "lca-mini-k6s2": "lca-base-dna6/vocab.txt",
        "lca-mini-k1s1": "lca-base-dna1/vocab.txt",
    }
}
def l2_norm(input, axis=1, epsilon=1e-12):
    norm = torch.norm(input, 2, axis, True)
    norm = torch.clamp(norm, min=epsilon)  # Avoid zero division
    output = torch.div(input, norm)
    return output

def initialize_linear_kaiming(layer: nn.Linear):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='linear')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

class ProkBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ProkBertModel`]. It is used to
    instantiate a ProkBert model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of a ProkBERT-base.

    Args:
        vocab_size (int, optional, defaults to ):
            Vocabulary size of the ProkBert model.
        hidden_size (int, optional, defaults to ):
            Dimension of the hidden representations.
        intermediate_size (int, optional, defaults to ):
            Dimension of the MLP representations.
        num_hidden_layers (int, optional, defaults to ):
            Number of hidden layers in the Transformer.
        num_attention_heads (int, optional, defaults to ):
            Number of attention heads for each attention layer.
        hidden_activation (str or function, optional, defaults to "gelu"):
            The activation function for intermediate layers.
        max_position_embeddings (int, optional, defaults to 8192):
            Maximum sequence length that this model might ever be used with.
        initializer_range (float, optional, defaults to 0.02):
            Standard deviation of the truncated_normal_initializer.
        initializer_cutoff_factor (float, optional, defaults to 2.0):
            The cutoff factor for weight initialization.
        norm_eps (float, optional, defaults to 1e-05):
            Epsilon for layer normalization.
        norm_bias (bool, optional, defaults to False):
            Whether to use bias in the normalization layers.
        pad_token_id (int, optional, defaults to 50283):
            Padding token id.
        eos_token_id (int, optional, defaults to 50282):
            End-of-sequence token id.
        bos_token_id (int, optional, defaults to 50281):
            Beginning-of-sequence token id.
        cls_token_id (int, optional, defaults to 50281):
            Classification token id.
        sep_token_id (int, optional, defaults to 50282):
            Separation token id.
        global_rope_theta (float, optional, defaults to 160000.0):
            Base period for the global rotary positional embeddings.
        attention_bias (bool, optional, defaults to False):
            Whether to use bias in query, key, value, and output projection layers.
        attention_dropout (float, optional, defaults to 0.0):
            Dropout rate for the attention probabilities.
        global_attn_every_n_layers (int, optional, defaults to 1):
            Set to 1 to use global attention in every layer (i.e. no sliding-window/local attention).
        local_attention (int, optional, defaults to 128):
            Window size for local attention (unused if `global_attn_every_n_layers` is 1).
        local_rope_theta (float, optional, defaults to 10000.0):
            Base period for the local rotary positional embeddings.
        embedding_dropout (float, optional, defaults to 0.0):
            Dropout rate applied to the embeddings.
        mlp_bias (bool, optional, defaults to False):
            Whether to use bias in the MLP layers.
        mlp_dropout (float, optional, defaults to 0.0):
            Dropout rate in the MLP layers.
        decoder_bias (bool, optional, defaults to True):
            Whether to use bias in the decoder layers.
        classifier_pooling (str, optional, defaults to "cls"):
            Pooling method for the classifier head, either "cls" or "mean".
        classifier_dropout (float, optional, defaults to 0.0):
            Dropout rate for the classifier head.
        classifier_bias (bool, optional, defaults to False):
            Whether to use bias in the classifier head.
        classifier_activation (str, optional, defaults to "gelu"):
            Activation function for the classifier head.
        deterministic_flash_attn (bool, optional, defaults to False):
            Whether to use deterministic flash attention.
        sparse_prediction (bool, optional, defaults to False):
            Whether to use sparse prediction for masked language modeling.
        sparse_pred_ignore_index (int, optional, defaults to -100):
            Index to ignore for sparse prediction.
        reference_compile (bool, optional):
            Whether to compile model layers for performance (if supported).
        repad_logits_with_grad (bool, optional, defaults to False):
            If True, logits are repadded with gradient tracking.
    """

    model_type = "prokbert"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 4608,
        hidden_size: int = 384,
        intermediate_size: int = 1152,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        hidden_activation: str = "gelu",
        max_position_embeddings: int = 16384,
        initializer_range: float = 0.02,
        initializer_cutoff_factor: float = 2.0,
        norm_eps: float = 1e-6,
        norm_bias: bool = False,
        pad_token_id: int = 0,
        eos_token_id: int = 3,
        bos_token_id: int = 2,
        cls_token_id: int = 2,
        sep_token_id: int = 3,
        global_rope_theta: float = 160000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        global_attn_every_n_layers: int = 1,  # Use global attention in every layer.
        local_attention: int = 256,
        local_rope_theta: float = 10000.0,
        embedding_dropout: float = 0.0,
        mlp_bias: bool = False,
        mlp_dropout: float = 0.0,
        decoder_bias: bool = True,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout: float = 0.0,
        classifier_bias: bool = False,
        classifier_activation: str = "gelu",
        deterministic_flash_attn: bool = False,
        sparse_prediction: bool = False,
        sparse_pred_ignore_index: int = -100,
        reference_compile: bool = None,
        repad_logits_with_grad: bool = False,
        norm_type: str = "rms",
        num_labels: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.global_rope_theta = global_rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_attention = local_attention
        self.local_rope_theta = local_rope_theta
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.decoder_bias = decoder_bias
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile
        self.repad_logits_with_grad = repad_logits_with_grad
        self.norm_type = norm_type
        self.num_labels = num_labels

        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {self.classifier_pooling}.'
            )

class ProkBertConfigCurr(ProkBertConfig):
    model_type = "prokbert"

    def __init__(
        self,
        bert_base_model = "neuralbioinfo/prokbert-mini",
        curricular_face_m = 0.5,
        curricular_face_s=64.,
        curricular_num_labels = 2,
        curriculum_hidden_size = -1, 
        classification_dropout_rate = 0.0, 
        **kwargs,
    ):
        super().__init__( **kwargs)
        self.curricular_num_labels = curricular_num_labels
        self.curricular_face_m = curricular_face_m
        self.curricular_face_s = curricular_face_s
        self.classification_dropout_rate = classification_dropout_rate
        self.bert_base_model = bert_base_model
        self.curriculum_hidden_size = curriculum_hidden_size

_CHECKPOINT_FOR_DOC = "example/prokbert-base"
_CONFIG_FOR_DOC = "ProkBertConfig"

PROK_BERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads, etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch module and refer to the PyTorch documentation for general usage and behavior.

    Parameters:
        config ([`ProkBertConfig`]):
            Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the model weights; see [`PreTrainedModel.from_pretrained`] for weight loading.
"""
#from prokbert2_config import *

# add near imports
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # root-mean-square normalization over last dim
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        if self.bias is not None:
            x = x + self.bias
        return self.weight * x
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        cos (torch.Tensor): The cosine part of the rotary embedding.
        sin (torch.Tensor): The sine part of the rotary embedding.
        position_ids (torch.Tensor, optional): Deprecated and unused.
        unsqueeze_dim (int, optional): The dimension along which to unsqueeze cos and sin.
    Returns:
        tuple(torch.Tensor): The rotated query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: "ProkBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # Apply rotary positional embedding to query and key.
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask

    # Upcast attention to fp32 for stability.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def flash_attention_forward(
    module: "ProkBertAttention",
    qkv: torch.Tensor,
    rotary_emb: "ProkBertUnpaddedRotaryEmbedding",
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    # qkv: (total_seqlen, 3, nheads, headdim)
    qkv = rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
    if convert_dtype:
        orig_dtype = qkv.dtype
        qkv = qkv.to(target_dtype)
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
        )
        attn = attn.to(orig_dtype)
    else:
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
        )
    return (attn.view(bs, dim),)


def sdpa_attention_forward(
    module: "ProkBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_output = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=module.attention_dropout if module.training else 0.0,
            attn_mask=attention_mask,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


PROK_BERT_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


def _unpad_prokbert_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Remove padding from input sequences.

    Args:
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), where 1 means valid and 0 means padding.
        position_ids: (batch, seqlen), optional position ids.
        labels: (batch, seqlen), optional labels.

    Returns:
        unpadded_inputs: Tensor of shape (total_nnz, ...) containing only valid tokens.
        indices: Tensor of indices corresponding to valid tokens.
        cu_seqlens: Cumulative sequence lengths of the unpadded tokens (shape: batch + 1).
        max_seqlen_in_batch: Maximum sequence length among all sequences (excluding padding).
        unpadded_position_ids: (total_nnz,) or None.
        unpadded_labels: (total_nnz,) or None.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels


def _pad_prokbert_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """
    Add padding back to the output tensor.

    Args:
        inputs: Tensor of shape (total_nnz, ...) containing outputs for only valid tokens.
        indices: Tensor of indices indicating positions of valid tokens.
        batch: Batch size.
        seqlen: Maximum sequence length (including padding).

    Returns:
        Tensor of shape (batch, seqlen, ...) with outputs in their original padded positions.
    """
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)
    return padded_inputs



class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # qkv: (total_nnz, 3, nheads, headdim)
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        # Combine the (3, nheads) dimensions for the first two channels to create a (total_nnz, 2*nheads, headdim) tensor.
        qk = qkv[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            qk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=False,
            inplace=True,
        )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
        dqk = do[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            dqk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=False,
            inplace=True,
            conjugate=True,
        )
        return do, None, None, None, None, None, None


def apply_rotary_unpadded(
    qkv,
    cos,
    sin,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Apply rotary embeddings to an unpadded (packed) QKV tensor.

    Args:
        qkv: Tensor of shape (total_nnz, 3, nheads, headdim) for packed QKV.
        cos, sin: Precomputed cosine and sine caches.
        cu_seqlens: Cumulative sequence lengths (batch + 1,).
        max_seqlen: Maximum sequence length in the batch.
    Returns:
        Tensor with rotary embeddings applied.
    """
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class ProkBertUnpaddedRotaryEmbedding(RotaryEmbedding):
    """
    Rotary embeddings for unpadded (packed) sequences used in ProkBERT.
    """

    def __init__(
        self,
        dim: int,
        base: float = 16000.0,
        max_seqlen: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            dim: Dimension of each head.
            base: Base for the rotary frequency computation.
            max_seqlen: Maximum sequence length to precompute the cosine and sine cache.
            device: Device on which to create the cache.
            dtype: Data type for the cache.
        """
        #super().__init__(dim=dim, base=base, pos_idx_in_fp32=True, device=device, interleaved=False)
        super().__init__(dim=dim, base=base, device=device, interleaved=False)

        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply rotary embeddings *inplace* to a packed QKV tensor.

        Args:
            qkv: Tensor of shape (total_nnz, 3, nheads, headdim).
            cu_seqlens: Cumulative sequence lengths tensor (batch + 1,).
            max_seqlen: Maximum sequence length in the current batch.
        Returns:
            Tensor with rotary embeddings applied.
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_unpadded(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"


class ProkBertEmbeddings(nn.Module):
    """
    Construct the embeddings from token embeddings, layer normalization, and dropout.
    """

    def __init__(self, config: ProkBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if config.norm_type == "rms":
            self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = nn.Dropout(config.embedding_dropout)

    @torch.compile(dynamic=True)
    def compiled_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))

    def forward(
        self, input_ids: torch.LongTensor = None, inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the embeddings layer.
        Args:
            input_ids: Tensor of input token ids.
            inputs_embeds: Alternatively, a pre-computed embedding tensor.
        Returns:
            Tensor of embeddings with normalization and dropout applied.
        """
        if inputs_embeds is not None:
            hidden_states = self.drop(self.norm(inputs_embeds))
        else:
            hidden_states = (
                self.compiled_embeddings(input_ids)
                if self.config.reference_compile
                else self.drop(self.norm(self.tok_embeddings(input_ids)))
            )
        return hidden_states




class ProkBertRotaryEmbedding(nn.Module):
    def __init__(self, config: ProkBertConfig, dim: int, base: float, device: Optional[torch.device] = None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(None, device, dim=dim, base=base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        Dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - Growing beyond the cached sequence length (allow scaling)
        2 - The current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class ProkBertMLP(nn.Module):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: ProkBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))
    

class ProkBertAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is available, this module uses it to improve throughput.
    Otherwise, it falls back on PyTorch's SDPA (or eager) implementation.
    """

    def __init__(self, config: ProkBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        max_position_embeddings = config.max_position_embeddings
        if self.local_attention != (-1, -1):
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta
            max_position_embeddings = config.local_attention

        if config._attn_implementation == "flash_attention_2":
            self.rotary_emb = ProkBertUnpaddedRotaryEmbedding(
                dim=self.head_dim, max_seqlen=max_position_embeddings, base=rope_theta
            )
        else:
            self.rotary_emb = ProkBertRotaryEmbedding(config=config, dim=self.head_dim, base=rope_theta)

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)
        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        attn_outputs = PROK_BERT_ATTENTION_FUNCTION[self.config._attn_implementation](
            self,
            qkv=qkv,
            rotary_emb=self.rotary_emb,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))
        return (hidden_states,) + attn_outputs[1:]
    


class test__ProkBertEncoderLayer(nn.Module):
    def __init__(self, config: ProkBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        # For the first layer, use Identity; otherwise, apply LayerNorm.
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            if config.norm_type == "rms":
                self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
            else:
                self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ProkBertAttention(config=config, layer_id=layer_id)
        if config.norm_type == "rms":
            self.mlp_norm  = RMSNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        else:
            self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ProkBertMLP(config)  # Assume you have a ProkBertMLP defined similarly.

class ProkBertEncoderLayer(nn.Module):
    def __init__(self, config: ProkBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config

        # choose norm kind once
        Norm = RMSNorm if config.norm_type == "rms" else nn.LayerNorm

        # Pre-LN everywhere (no layer_id==0 Identity)
        self.attn_norm = Norm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp_norm  = Norm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

        self.attn = ProkBertAttention(config=config, layer_id=layer_id)

        # MLP must exist regardless of norm type
        self.mlp = ProkBertMLP(config)


    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=output_attentions,
        )
        # Residual connection for attention.
        hidden_states = hidden_states + attn_outputs[0]
        # Apply the MLP block.
        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )
        hidden_states = hidden_states + mlp_output

        return (hidden_states,) + attn_outputs[1:]  # Return additional outputs (e.g. attentions) if provided.


PROK_BERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads, etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch module and refer to the PyTorch documentation for general usage and behavior.

    Parameters:
        config ([`ProkBertConfig`]):
            Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the model weights; see [`PreTrainedModel.from_pretrained`] for weight loading.
"""


@add_start_docstrings(
    "The bare ProkBert Model outputting raw hidden-states without any specific head on top.",
    PROK_BERT_START_DOCSTRING,
)
class ProkBertPreTrainedModel(PreTrainedModel):
    config_class = ProkBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ProkBertEmbeddings", "ProkBertEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False

    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size ** -0.5,
        }

        if isinstance(module, ProkBertEmbeddings):
            init_weight(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, ProkBertMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ProkBertAttention):
            init_weight(module.Wqkv, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ProkBertPredictionHead):
            init_weight(module.dense, stds["out"])
        elif isinstance(module, ProkBertForMaskedLM):
            init_weight(module.decoder, stds["out"])


    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        if config._attn_implementation_internal is None:
            config._attn_implementation_internal = "flash_attention_2"
            try:
                return cls._check_and_enable_flash_attn_2(
                    config,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    hard_check_only=False,
                    check_device_map=check_device_map,
                )
            except (ValueError, ImportError):
                config._attn_implementation_internal = None
        return super()._autoset_attn_implementation(
            config,
            use_flash_attention_2=use_flash_attention_2,
            torch_dtype=torch.float16,
            device_map=device_map,
            check_device_map=check_device_map,
        )

    def _maybe_set_compile(self):
        if self.config.reference_compile is False:
            return

        if hasattr(self, "hf_device_map") and len(self.hf_device_map) > 1:
            if self.config.reference_compile:
                logger.warning_once(
                    "If `accelerate` split the model across devices, `torch.compile` will not work. Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "mps":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.mps` device is not supported. Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "cpu":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.cpu` device is not supported. Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.config.reference_compile is None:
            self.config.reference_compile = is_triton_available()

    def resize_token_embeddings(self, *args, **kwargs):
        model_embeds = super().resize_token_embeddings(*args, **kwargs)

        if self.config.reference_compile in {True, None}:
            if self.config.reference_compile:
                logger.warning_once(
                    "Resizing token embeddings with `torch.compile` is not supported. Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        return model_embeds    
    
@add_start_docstrings(
    "The bare ProkBert Model outputting raw hidden-states without any specific head on top.",
    PROK_BERT_START_DOCSTRING,
)
class ProkBertModel(ProkBertPreTrainedModel):
    def __init__(self, config: ProkBertConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ProkBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ProkBertEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        if config.norm_type == "rms":
            self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        else:
            self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    #@add_start_docstrings_to_model_forward(PROK_BERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutput]:
        # Set defaults for outputs
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ensure exactly one of input_ids or inputs_embeds is provided.
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        repad = False
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_prokbert_input(
                            inputs=input_ids, attention_mask=attention_mask
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, *_ = _unpad_prokbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask
                    )
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            attention_mask, sliding_window_mask = self._update_attention_mask(
                attention_mask, output_attentions=output_attentions
            )

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    sliding_window_mask,
                    position_ids,
                    cu_seqlens,
                    max_seqlen,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if repad:
            hidden_states = _pad_prokbert_output(inputs=hidden_states, indices=indices, batch=batch_size, seqlen=seq_len)
            if all_hidden_states is not None:
                all_hidden_states = tuple(
                    _pad_prokbert_output(inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len)
                    for hs in all_hidden_states
                )

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _update_attention_mask(self, attention_mask: torch.Tensor, output_attentions: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if output_attentions:
            if self.config._attn_implementation == "sdpa":
                logger.warning_once(
                    "Outputting attentions is only supported with the 'eager' attention implementation, "
                    'not with "sdpa". Falling back to `attn_implementation="eager"`.'
                )
                self.config._attn_implementation = "eager"
            elif self.config._attn_implementation != "eager":
                logger.warning_once(
                    "Outputting attentions is only supported with the eager attention implementation, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`. '
                    "Setting `output_attentions=False`."
                )
        global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        distance = torch.abs(rows - rows.T)
        window_mask = ((distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0)
                       .to(attention_mask.device))
        sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min)
        return global_attention_mask, sliding_window_mask


class no_rms_cjoice_ProkBertPredictionHead(nn.Module):
    """
    ProkBertPredictionHead applies a dense layer followed by an activation function and layer normalization.
    This block is used as a preprocessing step before the final vocabulary projection in the masked language modeling head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)



class ProkBertPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        Norm = RMSNorm if getattr(config, "norm_type", "layernorm") == "rms" else nn.LayerNorm
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = Norm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)




    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Applies the dense projection, activation, and normalization.
        return self.norm(self.act(self.dense(hidden_states)))


@add_start_docstrings(
    "The ProkBert Model with a decoder head on top that is used for masked language modeling.",
    PROK_BERT_START_DOCSTRING,
)

class ProkBertForMaskedLM(ProkBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = ProkBertModel(config)  
        self.head = ProkBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing.
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    #@add_start_docstrings_to_model_forward(PROK_BERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        # For flash attention, unpad the inputs to avoid wasting compute on padding tokens.
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_prokbert_input(
                            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_prokbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                    )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        # If sparse prediction is enabled, filter out non-masked tokens.
        if self.sparse_prediction and labels is not None:
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.head(last_hidden_state))
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

        # If using flash attention, repad the output.
        if self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_prokbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m=0.5, s=64.):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        initialize_linear_kaiming(self.kernel)

    def forward(self, embeddings, label):
        # Normalize embeddings and the classifier kernel
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # Compute cosine similarity between embeddings and kernel columns
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        
        # Clone original cosine values (used later for analysis if needed)
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        
        # Get the cosine values corresponding to the ground-truth classes
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), label].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target + margin)
        
        # Create a mask for positions where the cosine similarity exceeds the modified value
        mask = (cos_theta > cos_theta_m) #.to(dtype=torch.uint8)
        
        # Apply the margin condition: for values greater than threshold, use cosine with margin;
        # otherwise subtract a fixed term.
        final_target_logit = torch.where(target_logit > self.threshold, 
                                         cos_theta_m, 
                                         target_logit - self.mm)
        
        # Update the buffer 't' (used to control the weight of hard examples)
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        
        # For the positions in the mask, re-scale the logits
        try:
            hard_example = cos_theta[mask]
        except Exception as e:
            print("Label max")
            print(torch.max(label))
            print("Shapes:")
            print(embeddings.shape)
            print(label.shape)
            hard_example = cos_theta[mask]
            
        cos_theta[mask] = hard_example * (self.t + hard_example)
        
        # Replace the logits of the target classes with the modified target logit
        final_target_logit = final_target_logit.to(cos_theta.dtype)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s

class ProkBertForCurricularClassification(ProkBertPreTrainedModel):
    config_class = ProkBertConfigCurr
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #bert_config = AutoConfig.from_pretrained(config.bert_base_model)
        #self.bert = ProkBertModel.from_pretrained(config.bert_base_model)

        #self.bert = ProkBertModel(config)
        self.model = ProkBertModel(config)

        
        # A weighting layer for pooling the sequence output
        self.weighting_layer = nn.Linear(self.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.config.classification_dropout_rate)

        if config.curriculum_hidden_size != -1:
            self.linear = nn.Linear(self.config.hidden_size, config.curriculum_hidden_size)
            initialize_linear_kaiming(self.linear)
            # Replace the simple classifier with the CurricularFace head.
            # Defaults m=0.5 and s=64 are used, but these can be adjusted if needed.
            self.curricular_face = CurricularFace(config.curriculum_hidden_size, 
                                                self.config.curricular_num_labels,
                                                m=self.config.curricular_face_m,
                                                s=self.config.curricular_face_s)
        else:
            self.linear = nn.Identity()
            self.curricular_face = CurricularFace(self.config.hidden_size, 
                                                self.config.curricular_num_labels,
                                                m=self.config.curricular_face_m,
                                                s=self.config.curricular_face_s)
        

        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get the outputs from the base ProkBert model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)
        
        # Pool the sequence output using a learned weighting (attention-like)
        # Compute raw weights
        weights = self.weighting_layer(sequence_output).squeeze(-1)  # (batch_size, seq_length)

        # Ensure mask shape matches
        if attention_mask.dim() == 2:
            mask = attention_mask
        elif attention_mask.dim() == 4:
            mask = attention_mask.squeeze(1).squeeze(1)  # (batch_size, seq_length)
        else:
            raise ValueError(f"Unexpected attention_mask shape {attention_mask.shape}")

        # Apply mask (masked positions -> -inf before softmax)
        weights = weights.masked_fill(mask == 0, float('-inf'))

        # Normalize
        weights = torch.nn.functional.softmax(weights, dim=1)  # (batch_size, seq_length)

        # Weighted pooling
        weights = weights.unsqueeze(-1)                        # (batch_size, seq_length, 1)
        print(weights.shape)
        print(sequence_output.shape)
        pooled_output = torch.sum(weights * sequence_output, dim=1)  # (batch_size, hidden_size)
        print(pooled_output.shape)
        # Classifier head
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear(pooled_output)
        print(pooled_output.shape)

        # CurricularFace requires the embeddings and the corresponding labels.
        # Note: During inference (labels is None), we just return l2 norm of bert part of the model
        if labels is None:
            return l2_norm(pooled_output, axis = 1) 
        else:
            logits, origin_cos = self.curricular_face(pooled_output, labels)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ProkBertForCurricularClassificationAvg(ProkBertPreTrainedModel):
    config_class = ProkBertConfigCurr
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #bert_config = AutoConfig.from_pretrained(config.bert_base_model)
        self.bert = ProkBertModel.from_pretrained(config.bert_base_model)
    
        self.dropout = nn.Dropout(self.config.classification_dropout_rate)

        if config.curriculum_hidden_size != -1:
            self.linear = nn.Linear(self.config.hidden_size, config.curriculum_hidden_size)
            initialize_linear_kaiming(self.linear)
            # Replace the simple classifier with the CurricularFace head.
            # Defaults m=0.5 and s=64 are used, but these can be adjusted if needed.
            self.curricular_face = CurricularFace(config.curriculum_hidden_size, 
                                                self.config.curricular_num_labels,
                                                m=self.config.curricular_face_m,
                                                s=self.config.curricular_face_s)
        else:
            self.linear = nn.Identity()
            self.curricular_face = CurricularFace(self.config.hidden_size, 
                                                self.config.curricular_num_labels,
                                                m=self.config.curricular_face_m,
                                                s=self.config.curricular_face_s)
        

<<<<<<< HEAD
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get the outputs from the base ProkBert model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
=======

class ProkBertForSequenceClassification(ProkBertPreTrainedModel):
    """
    ProkBERT model for sequence classification tasks.
    """
    def __init__(self, config: ProkBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ProkBertModel(config)           
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-5)

        self.weighting_layer = nn.Linear(self.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()

        #print("fsgfgdfgfd")
        #print(self.num_labels)
        #print(self.config)
        #print(self.config.problem_type)

        # Base ProkBERT model
        #self.model = ProkBertModel(config)
        # Intermediate head for classification pooling
        #self.head = ProkBertPredictionHead(config)
        # Dropout and final classifier
        #self.dropout = nn.Dropout(config.classifier_dropout)
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=config.classifier_bias)

        # Initialize weights and apply final processing
        self.post_init()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)

        # Uniform token attention at start
        if module is self.weighting_layer:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

        # BERT-style head init: small Gaussian, zero bias
        if module is self.classifier:
            # both are good:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            module.weight.data /= math.sqrt(self.classifier.in_features)  # extra shrink
            nn.init.zeros_(module.bias)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        sliding_window_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        indices: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        batch_size: int = None,
        seq_len: int = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        # Determine return type
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        #print('Getting the input ids:')
        #print(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
>>>>>>> 8d02c79 (Adding RMS norm, new classification model for modernbert, changing loss function to KL)
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
<<<<<<< HEAD
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)

        # Ensure mask shape matches (batch_size, seq_length)
        if attention_mask.dim() == 2:
            mask = attention_mask
        elif attention_mask.dim() == 4:
            mask = attention_mask.squeeze(1).squeeze(1)  # (batch_size, seq_length)
        else:
            raise ValueError(f"Unexpected attention_mask shape {attention_mask.shape}")

        # Expand mask to match hidden size
        mask_expanded = mask.unsqueeze(-1).expand(sequence_output.size())  # (batch_size, seq_length, hidden_size)

        # Apply mask and compute mean pooling
        sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)  # (batch_size, hidden_size)
        lengths = torch.clamp(mask.sum(dim=1, keepdim=True), min=1e-9).to(sequence_output.dtype)      # avoid division by zero
        pooled_output = (sum_embeddings / lengths).to(sequence_output.dtype)                           # (batch_size, hidden_size)
        # Classifier head
        #print(pooled_output.dtype)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear(pooled_output)

        # CurricularFace requires the embeddings and the corresponding labels.
        # Note: During inference (labels is None), we just return l2 norm of bert part of the model
        if labels is None:
            return l2_norm(pooled_output, axis = 1) 
        else:
            logits, origin_cos = self.curricular_face(pooled_output, labels)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.view(-1))
        
=======
        # Get hidden states
        sequence_output = outputs[0]
        weights = self.weighting_layer(sequence_output)
        weights = torch.nn.functional.softmax(weights, dim=1)            
        # Compute weighted sum
        pooled_output = torch.sum(weights * sequence_output, dim=1)   
        pooled_output = self.norm(pooled_output)          
        # Classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #print(logits)
        #1/0

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        classification_output = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return classification_output
        print(last_hidden_state)


class ddssProkBertForSequenceClassification(ProkBertPreTrainedModel):
    """
    ProkBERT model for sequence classification tasks.
    """
    def __init__(self, config: ProkBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        #print("fsgfgdfgfd")
        #print(self.num_labels)
        #print(self.config)
        #print(self.config.problem_type)

        # Base ProkBERT model
        self.model = ProkBertModel(config)
        # Intermediate head for classification pooling
        self.head = ProkBertPredictionHead(config)
        # Dropout and final classifier
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=config.classifier_bias)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: nn.Module):
        # first let the base class init everything else
        super()._init_weights(module)

        # then catch our pooling head and zero it
        if module is getattr(self, "weighting_layer", None):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        sliding_window_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        indices: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        batch_size: int = None,
        seq_len: int = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        # Determine return type
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        print('Getting the input ids:')
        print(input_ids)

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Get hidden states
        last_hidden_state = outputs[0]
        #print('Last hidden state: ')
        #print(last_hidden_state)

        # Pooling
        if self.config.classifier_pooling == "cls":
            pooled = last_hidden_state[:, 0]
        else:
            # mean pooling over valid tokens
            pooled = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        print(pooled)
        # Classification head
        pooled = self.head(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            # Determine problem type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            print('Guessed problem type:' + self.config.problem_type)
            # Compute loss
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #print('Loss')
        #print(loss)

        #1/0

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

>>>>>>> 8d02c79 (Adding RMS norm, new classification model for modernbert, changing loss function to KL)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

<<<<<<< HEAD
class ProkBertForSequenceClassification(ProkBertPreTrainedModel):
=======



class ddssProkBertForSequenceClassification(ProkBertPreTrainedModel):
>>>>>>> 8d02c79 (Adding RMS norm, new classification model for modernbert, changing loss function to KL)
    """
    ProkBERT model for sequence classification tasks.
    """
    def __init__(self, config: ProkBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        print("fsgfgdfgfd")
        print(self.num_labels)
        print(self.config)
        print(self.config.problem_type)

        # Base ProkBERT model
        self.model = ProkBertModel(config)
        # Intermediate head for classification pooling
        self.head = ProkBertPredictionHead(config)
        # Dropout and final classifier
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=config.classifier_bias)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: nn.Module):
        # first let the base class init everything else
        super()._init_weights(module)

        # then catch our pooling head and zero it
        if module is getattr(self, "weighting_layer", None):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        sliding_window_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        indices: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        batch_size: int = None,
        seq_len: int = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        # Determine return type
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Get hidden states
        last_hidden_state = outputs[0]

        # Pooling
        if self.config.classifier_pooling == "cls":
            pooled = last_hidden_state[:, 0]
        else:
            # mean pooling over valid tokens
            pooled = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        # Classification head
        pooled = self.head(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            # Determine problem type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            #print('Guessed problem type:' + self.config.problem_type)
            # Compute loss
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class ProkBertForMaskedLM2(ProkBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model   = ProkBertModel(config)
        self.head    = ProkBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=config.decoder_bias)

        # for sparse‐integer masking (legacy)
        self.sparse_prediction       = config.sparse_prediction
        self.sparse_pred_ignore_index = config.sparse_pred_ignore_index

        # finish init
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @torch.compile(dynamic=True)
    def compiled_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(hidden))

    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor]    = None,
        attention_mask: Optional[torch.Tensor]    = None,
        sliding_window_mask: Optional[torch.Tensor]= None,
        position_ids: Optional[torch.Tensor]      = None,
        inputs_embeds: Optional[torch.Tensor]     = None,
        labels: Optional[torch.LongTensor]        = None,
        labels_dist: Optional[torch.FloatTensor]  = None,
        loss_mask: Optional[torch.BoolTensor]     = None,
        indices: Optional[torch.Tensor]           = None,
        cu_seqlens: Optional[torch.Tensor]        = None,
        max_seqlen: Optional[int]                 = None,
        batch_size: Optional[int]                 = None,
        seq_len: Optional[int]                    = None,
        output_attentions: Optional[bool]         = None,
        output_hidden_states: Optional[bool]      = None,
        return_dict: Optional[bool]               = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # debug inputs
        '''
        print("Input IDs:", input_ids.shape); print(input_ids); print("——")
        if labels_dist is not None:
            print("Labels dist:", labels_dist.shape); print(labels_dist); print("——")
        if loss_mask is not None:
            print("Loss mask:", loss_mask.shape); print(loss_mask); print("——")
        '''
        #print('___________')
        #print('Labels:')
        #print(labels)
        #print('___________')

        # 1) Optional unpad for flash_attention_2
        if self.config._attn_implementation == "flash_attention_2" \
        and indices is None and cu_seqlens is None and max_seqlen is None:
            # infer batch_size, seq_len
            if batch_size is None or seq_len is None:
                if inputs_embeds is not None:
                    batch_size, seq_len = inputs_embeds.shape[:2]
                else:
                    batch_size, seq_len = input_ids.shape[:2]
            # EXPLICIT device pick
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            attention_mask = attention_mask if attention_mask is not None else \
                            torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
            if inputs_embeds is None:
                with torch.no_grad():
                    input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = \
                        _unpad_prokbert_input(
                            inputs=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=labels
                        )
            else:
                inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = \
                    _unpad_prokbert_input(
                        inputs=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        labels=labels
                    )

        # 2) Core encoder
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # (B,L,H) or packed (N,H)
        #print('outputs:')
        #print(outputs)

        # 3) Legacy sparse integer mask
        if self.sparse_prediction and labels is not None:
            #print('Sparse predictions..')
            flat_labels = labels.view(-1)
            flat_hidden = sequence_output.view(flat_labels.shape[0], -1)
            mask_tokens = flat_labels != self.sparse_pred_ignore_index
            sequence_output = flat_hidden[mask_tokens]
            labels = flat_labels[mask_tokens]

        # 4) Project to vocab
        if self.config.reference_compile:
            logits = self.compiled_head(sequence_output)
        else:
            hidden = self.head(sequence_output)
            logits = self.decoder(hidden)
        #print("Raw logits shape:", logits.shape); print("——")

        loss = None
        V    = self.config.vocab_size

        # 5a) Integer‐label MLM
        if labels is not None:
            #print('Using the original stuff!')
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, V), labels.view(-1))
            #print(f'Loss: {loss}')

        # 5b) Soft‐distribution MLM (no re‐pad)
        elif labels_dist is not None and loss_mask is not None:
            B, L = loss_mask.shape
            flat_mask = loss_mask.view(-1)        # (B*L,)
            flat_dist = labels_dist.view(-1, V)   # (B*L, V)

            # packed by attention_mask
            if logits.dim() == 2 and logits.shape[0] != flat_mask.sum().item():
                full_attn    = attention_mask.view(-1)     # (B*L,)
                assert logits.shape[0] == full_attn.sum().item()
                dist_attn    = flat_dist[full_attn]        # (Natt, V)
                mask_in_attn = flat_mask[full_attn]        # (Natt,)
                pred = logits[mask_in_attn]                # (N_mask, V)
                targ = dist_attn[mask_in_attn]             # (N_mask, V)

            # packed exactly by loss_mask
            elif logits.dim() == 2 and logits.shape[0] == flat_mask.sum().item():
                pred = logits
                targ = flat_dist[flat_mask]

            # full (B,L,V)
            else:
                flat_logits = logits.view(-1, V)           # (B*L, V)
                pred        = flat_logits[flat_mask]       # (N_mask, V)
                targ        = flat_dist[flat_mask]         # (N_mask, V)

            #print("Packed pred.shape:", pred.shape)
            #print("Packed targ.shape:", targ.shape)
            #print("Sum targ rows:", targ.sum(dim=-1))
            eps  = 1e-8
            targ = targ.clamp_min(eps)
            targ = targ / targ.sum(dim=-1, keepdim=True)
            targ = targ.to(pred.dtype).detach()   

            #print('Tar')
            #print(targ)
            #print(targ.shape)            
            #print(targ[0,:])
            #print(targ[0,:].sum())
            logp = F.log_softmax(pred, dim=-1)
            loss = F.kl_div(logp, targ, reduction="batchmean")  

            #print(loss)
            #print('prevois loss: ')
            #print( -(targ * logp).sum(dim=-1).mean())

            #1/0
            #logp = F.log_softmax(pred, dim=-1)
            #loss = -(targ * logp).sum(dim=-1).mean()

        if self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_prokbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)

        # 6) Return
        if not return_dict:
            out = (logits,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out

        return MaskedLMOutput(
            loss=loss,
            logits=logits if logits.dim() == 3 else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
