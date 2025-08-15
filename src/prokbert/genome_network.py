from typing import Optional, Tuple, Literal, Union

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.utils import is_flash_attn_2_available, logging
from transformers.utils.import_utils import is_triton_available

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

from .models import ProkBertConfig, ProkBertModel


logger = logging.get_logger(__name__)


class GenomeNetworkConfig(PretrainedConfig):
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

    model_type = "genome_network"
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
        norm_eps: float = 1e-5,
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
        reference_compile: bool = False,
        repad_logits_with_grad: bool = False,
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
        self.tie_word_embeddings = False

        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError(f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {self.classifier_pooling}.')


def eager_attention_forward(
    module: "GenomeNetworkAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)

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
    module: "GenomeNetworkAttention",
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    # qkv: (total_seqlen, 3, nheads, headdim)
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
    module: "GenomeNetworkAttention",
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
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)

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
    inputs: torch.LongTensor,
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


class GenomeNetworkMLP(nn.Module):
    def __init__(self, config: GenomeNetworkConfig):
        super().__init__()

        self.config = config

        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class GenomeNetworkAttention(nn.Module):
    def __init__(self, config: GenomeNetworkConfig, layer_id: Optional[int] = None):
        super().__init__()

        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})")

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

        max_position_embeddings = config.max_position_embeddings # ? unused
        if self.local_attention != (-1, -1):
            max_position_embeddings = config.local_attention # ? unused

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
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))
        return (hidden_states,) + attn_outputs[1:]


class GenomeNetworkEncoderLayer(nn.Module):
    def __init__(self, config: GenomeNetworkConfig, layer_id: Optional[int] = None):
        super().__init__()

        self.config = config

        # For the first layer, use Identity; otherwise, apply LayerNorm.
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = GenomeNetworkAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = GenomeNetworkMLP(config)

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


class GenomeNetworkPreTrainedModel(PreTrainedModel):
    config_class = GenomeNetworkConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GenomeNetworkEmbeddings", "GenomeNetworkEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False

    def _init_weights(self, module: nn.Module):

        cutoff_factor = self.config.initializer_cutoff_factor

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

        if isinstance(module, GenomeNetworkMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, GenomeNetworkAttention):
            init_weight(module.Wqkv, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, GenomeNetworkPredictionHead):
            init_weight(module.dense, stds["out"])
        elif isinstance(module, GenomeNetworkLMPredictionHead):
            init_weight(module.decoder, stds["out"])

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


class GenomeNetworkModel(GenomeNetworkPreTrainedModel):
    def __init__(self, config: GenomeNetworkConfig, embeddings_model: ProkBertModel):
        super().__init__(config)
        self.config = config
        self.embeddings_model = embeddings_model
        self.layers = nn.ModuleList(
            [GenomeNetworkEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask_genom: Optional[torch.Tensor] = None,
        attention_mask_gene: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        gene_len: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...] | BaseModelOutput:

        # Set defaults for outputs
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You must specify input_ids!")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask_genom)

        assert input_ids.dim() == 3, f"Expected input_ids to be of shape (batch_size, gene_len, seq_len), but got {input_ids.shape}."
        if batch_size is None or seq_len is None or gene_len is None:
                batch_size, gene_len, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask_genom is None:
            attention_mask_genom = torch.ones((batch_size, gene_len), device=device, dtype=torch.bool)

        if attention_mask_gene is None:
            attention_mask_gene = torch.ones((gene_len, seq_len), device=device, dtype=torch.bool)

        repad = False
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                with torch.no_grad():
                    input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_prokbert_input( # ?
                        inputs=input_ids, attention_mask=attention_mask_genom
                    )
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            attention_mask, sliding_window_mask = self._update_attention_mask(  # ?
                attention_mask_genom, output_attentions=output_attentions
            )

        # TODO create separate method?
        embeddings = self.embeddings_model(
            input_ids=input_ids.reshape(-1, seq_len),
            attention_mask=attention_mask_gene.reshape(-1, seq_len),
            token_type_ids=token_type_ids.reshape(-1, seq_len) if token_type_ids is not None else None
        ).pooler_output
        hidden_states = embeddings.reshape(batch_size, gene_len, -1)

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

        if repad:  # ?
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


class GenomeNetworkPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        if isinstance(config.classifier_activation, str):
            self.act = ACT2FN[config.classifier_activation]
        else:
            self.act = config.classifier_activation
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GenomeNetworkLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.head = GenomeNetworkPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.head(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class GenomeNetworkForMaskedLM(GenomeNetworkPreTrainedModel):
    _tied_weights_keys = ["cls.decoder"]

    def __init__(
        self,
        config: GenomeNetworkConfig,
        embedding_model: Optional[PreTrainedModel] = None,
        embedding_config: Optional[ProkBertConfig] = None,
        **kwargs
    ) -> None:

        if not isinstance(config, GenomeNetworkConfig):
            raise ValueError(f"Expected `GenomeNetworkConfig`, got {config.__class__.__module__}.{config.__class__.__name__}")

        super().__init__(config, **kwargs)
        self.config = config

        if (embedding_model is None) == (embedding_config is None):
            raise ValueError("You must specify exactly one of embedding_model or embedding_config")

        if embedding_model is None:
            embedding_model = ProkBertModel(embedding_config)

        self.bert = GenomeNetworkModel(config, embedding_model)
        self.cls = GenomeNetworkLMPredictionHead(config)

        self.post_init() # initalize weights and apply final processing

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.cls.decoder(self.cls.head(output))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
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
    ) -> Union[tuple,  MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self._maybe_set_compile()

        # For flash attention, unpad the inputs to avoid wasting compute on padding tokens.
        # if self.config._attn_implementation == "flash_attention_2":
        #     if indices is None and cu_seqlens is None and max_seqlen is None:
        #         if batch_size is None and seq_len is None:
        #             if inputs_embeds is not None:
        #                 batch_size, seq_len = inputs_embeds.shape[:2]
        #             else:
        #                 batch_size, seq_len = input_ids.shape[:2]
        #         device = input_ids.device if input_ids is not None else inputs_embeds.device

        #         if attention_mask is None:
        #             attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        #         if inputs_embeds is None:
        #             with torch.no_grad():
        #                 input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_prokbert_input(
        #                     inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
        #                 )
        #         else:
        #             inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_prokbert_input(
        #                 inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels
        #             )

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
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

        # filter out non-masked tokens, decrease compute in head + loss
        if self.config.sparse_prediction and labels is not None:
            labels = labels.view(-1) # (BxS) <- (B, S)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1) # (BxS, H)
            mask_tokens = labels != self.config.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens] # (BxS, H) <- (B, S, H)
            labels = labels[mask_tokens] # (BxS)

        logits = ( # (BxS, V)
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.cls(last_hidden_state)
        )

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # If using flash attention, repad the output.
        # if self.config._attn_implementation == "flash_attention_2": # ?
        #     with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
        #         logits = _pad_prokbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
