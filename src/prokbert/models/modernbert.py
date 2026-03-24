from typing import Optional, Tuple, Union, Literal, Callable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers import initialization as init
from transformers.processing_utils import Unpack
from transformers.utils.auto_docstring import auto_docstring
from transformers.utils.output_capturing import capture_outputs
from transformers.models.align.modeling_align import eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.integrations import use_kernel_func_from_hub, use_kernelized_func
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput, MaskedLMOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters, dynamic_rope_update
from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
from transformers.utils.generic import TransformersKwargs, maybe_autocast, can_return_tuple, merge_with_config_defaults


logger = logging.get_logger(__name__)


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
        rope_parameters: (RopeParameters or dict, optional):
            Parameters for the RoPE positional embeddings.
            If not provided, default RoPE parameters will be computed based on the config.
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
            Argument is deprecated and will be removed in a future version.
    """

    model_type = "prokbert"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = {"global": 160_000.0, "local": 10_000.0}

    def __setattr__(self, name, value):
        if name == "reference_compile" and value is not None:
            logger.warning_once(
                "The `reference_compile` argument is deprecated and will be removed in `transformers v5.2.0`"
                "Use `torch.compile()` directly on the model instead."
            )
            value = None
        super().__setattr__(name, value)

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
        layer_types: list[str] | None = None,
        rope_parameters: dict[Literal["full_attention", "sliding_attention"], RopeParameters] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        local_attention: int = 256,
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
        reference_compile: bool | None = None,
        tie_word_embeddings: bool = True,
        norm_type: str = "rms",
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
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
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.local_attention = local_attention
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.decoder_bias = decoder_bias
        self.rope_parameters = rope_parameters
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile
        self.tie_word_embeddings = tie_word_embeddings
        self.norm_type = norm_type

        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {self.classifier_pooling}.'
            )

        if self.norm_type not in ["layernorm", "rms"]:
            raise ValueError(
                f'Invalid value for `norm_type`, should be either "layernorm" or "rms", but is {self.norm_type}.'
            )

        self.layer_types = layer_types

        self.global_attn_every_n_layers = kwargs.get("global_attn_every_n_layers", 1)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool(i % self.global_attn_every_n_layers) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        super().__init__(**kwargs)

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation: set | None = None, **kwargs):

        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params

        if self.rope_parameters.get("full_attention") is None:
            self.rope_parameters["full_attention"] = {"rope_type": "default"}
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("global_rope_theta", self.default_theta["global"])
        )
        if self.rope_parameters.get("sliding_attention") is None:
            self.rope_parameters["sliding_attention"] = {"rope_type": "default"}
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("local_rope_theta", self.default_theta["local"])
        )

        self.standardize_rope_params()
        # NOTE the self.validate_rope method will casues a warning:
        #   "Unrecognized keys in `rope_parameters` for 'rope_type'='default':
        #    {'sliding_attention', 'full_attention'}"
        # It can be safely ignored, as it comes from the fact we don't
        # use sliding_attention, which is set in `global_attn_every_n_layers`
        # later we might use it, so the sliding attention code parts should remain
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs

    def to_dict(self):
        output = super().to_dict()
        output.pop("reference_compile", None)
        return output

    @property
    def sliding_window(self):
        """Half-window size: `local_attention` is the total window, so we divide by 2."""
        return self.local_attention // 2

    @sliding_window.setter
    def sliding_window(self, value):
        """Set sliding_window by updating local_attention to 2 * value."""
        self.local_attention = value * 2


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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        if self.bias is not None:
            x = x + self.bias
        return self.weight * x


class ProkBertEmbeddings(nn.Module):
    def __init__(self, config: ProkBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if config.norm_type == "rms":
            self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(
        self, input_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = self.drop(self.norm(inputs_embeds))
        else:
            hidden_states = self.drop(self.norm(self.tok_embeddings(input_ids)))
        return hidden_states


class ProkBertMLP(nn.Module):
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


class ProkBertRotaryEmbedding(nn.Module):
    def __init__(self, config: ProkBertConfig, device: Optional[torch.device] = None):
        super().__init__()

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.layer_types = list(set(config.layer_types))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: ProkBertConfig,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:

        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # unused in this type of RoPE

        # compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    original_dtype = q.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(original_dtype), k_embed.to(original_dtype)


@use_kernelized_func(apply_rotary_pos_emb)
class ProkBertAttention(nn.Module):
    def __init__(self, config: ProkBertConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.head_dim: int = config.hidden_size // config.num_attention_heads
        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * self.head_dim * config.num_attention_heads, bias=config.attention_bias
        )

        if config.layer_types[layer_idx] == "sliding_attention":
            # config.sliding_window = local_attention // 2 (half-window size, e.g. 64 for local_attention=128)
            # +1 is needed because flash attention sets inclusive boundaries (see modeling_flash_attention_utils.py)
            self.sliding_window = config.sliding_window + 1
        else:
            self.sliding_window = None

        self.is_causal = False

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        input_shape = hidden_states.shape[:-1]

        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(*input_shape, 3, -1, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(dim=-3)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        # NOTE sdpa attention (the `sdpa_attention_forward` fn) calls pytorch's
        # `scaled_dot_product_attention` fn which requires boolean attention mask,
        # for other types, e.g. eager attention it is converted to float (zeros and -inf values)
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.head_dim**-0.5,
            sliding_window=self.sliding_window,
            deterministic=self.deterministic_flash_attn,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_drop(self.Wo(attn_output))
        return attn_output, attn_weights


class ProkBertEncoderLayer(nn.Module):
    def __init__(self, config: ProkBertConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config

        norm = RMSNorm if config.norm_type == "rms" else nn.LayerNorm

        self.attn_norm = norm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp_norm  = norm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ProkBertAttention(config=config, layer_idx=layer_idx)
        self.mlp = ProkBertMLP(config)

        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:

        attn_output, _ = self.attn(
            self.attn_norm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


@auto_docstring
class ProkBertPreTrainedModel(PreTrainedModel):
    config_class = ProkBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ProkBertEmbeddings", "ProkBertEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": ProkBertEncoderLayer,
        "attentions": ProkBertAttention,
    }

    @torch.no_grad()
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
        elif isinstance(
            module,
            (
                ProkBertForSequenceClassification,
            ),
        ):
            # uniform token attention at start
            nn.init.zeros_(module.weighting_layer.weight)
            nn.init.zeros_(module.weighting_layer.bias)

            # BERT-style head init: small gaussian, zero bias
            nn.init.xavier_uniform_(module.classifier.weight, gain=1.0)
            module.classifier.weight.data /= math.sqrt(module.classifier.in_features)  # extra shrink
            nn.init.zeros_(module.classifier.bias)

        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, ProkBertRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class ProkBertModel(ProkBertPreTrainedModel):
    def __init__(self, config: ProkBertConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ProkBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ProkBertEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.norm_type == "rms":
            self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        else:
            self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.rotary_emb = ProkBertRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
       self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutput:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        if not isinstance(attention_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": hidden_states,
                # NOTE if not provided, create_bidirectional_mask will default to
                # full attention, which will be None (full attention doesn't require a mask)
                "attention_mask": attention_mask,
            }
            attention_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }

        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask_mapping[encoder_layer.attention_type],
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class ProkBertPredictionHead(nn.Module):
    def __init__(self, config: ProkBertConfig):
        super().__init__()
        norm = RMSNorm if getattr(config, "norm_type", "layernorm") == "rms" else nn.LayerNorm
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = norm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


@auto_docstring(
    custom_intro="""
    The ProkBert Model with a decoder head on top that is used for masked language modeling.
    """
)
class ProkBertForMaskedLM(ProkBertPreTrainedModel):
    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: ProkBertConfig):
        super().__init__(config)
        self.config = config
        self.model = ProkBertModel(config)
        self.head = ProkBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = config.sparse_prediction
        self.sparse_pred_ignore_index = config.sparse_pred_ignore_index

        # initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        labels_dist: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        last_hidden_state = outputs[0]  # (batch_size, seq_len, hidden_size)

        if self.sparse_prediction:

            if labels is not None:
                # flatten labels and last hidden state (output)
                labels = labels.view(-1)
                last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

                # then filter out the non-masked tokens
                mask_tokens = labels != self.sparse_pred_ignore_index
                last_hidden_state = last_hidden_state[mask_tokens]
                labels = labels[mask_tokens]

            # batch_size x seq_len x probability distribution over ACGT nucleotids (4 items)
            elif labels_dist is not None:
                # flatten target distribution and last hidden state (output)
                labels_dist = labels_dist.view(-1, self.config.vocab_size)
                last_hidden_state = last_hidden_state.view(labels_dist.shape[0], -1)

                # extract only the masked tokens (where labels_dist is not all zeros)
                mask_tokens = labels_dist.sum(dim=-1) != 0
                last_hidden_state = last_hidden_state[mask_tokens]
                labels_dist = labels_dist[mask_tokens]

        logits = self.decoder(self.head(last_hidden_state))

        loss = None
        # standard MLM loss (cross-entropy)
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)
        # soft-distribution MLM loss (KL divergence)
        elif labels_dist is not None:
            # normalize target distribution
            labels_dist = labels_dist.clamp_min(1e-8)
            labels_dist = labels_dist / labels_dist.sum(dim=-1, keepdim=True)
            labels_dist = labels_dist.to(logits.dtype).detach()

            logits = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(logits, labels_dist, reduction="batchmean")

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ProkBert Model with a sequence classification head on top for sequence classification tasks.
    """
)
class ProkBertForSequenceClassification(ProkBertPreTrainedModel):
    def __init__(self, config: ProkBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ProkBertModel(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.weighting_layer = nn.Linear(self.config.hidden_size, 1)
        self.drop = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        # initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        last_hidden_state = outputs[0]  # (batch_size, seq_len, hidden_size)

        weights = self.weighting_layer(last_hidden_state) # (batch_size, seq_len, 1)
        weights = torch.nn.functional.softmax(weights, dim=1)

        pooled_output = torch.sum(weights * last_hidden_state, dim=1) # (batch_size, hidden_size)
        pooled_output = self.norm(pooled_output)

        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None: # single label classification
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ProkBertConfig",
    "ProkBertModel",
    "ProkBertPreTrainedModel",
    "ProkBertForMaskedLM",
    "ProkBertForSequenceClassification",
]
