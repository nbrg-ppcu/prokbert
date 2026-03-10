
import inspect
import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
except ImportError:  # pragma: no cover - compatibility fallback for older Transformers
    class Cache:  # type: ignore[no-redef]
        pass

    class DynamicCache(Cache):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            super().__init__()

        def get_seq_length(self):
            return 0

    class EncoderDecoderCache(Cache):  # type: ignore[no-redef]
        def __init__(self, self_attention_cache=None, cross_attention_cache=None):
            super().__init__()
            self.self_attention_cache = self_attention_cache
            self.cross_attention_cache = cross_attention_cache
            self.is_updated = {}

        @classmethod
        def from_legacy_cache(cls, past_key_values):
            cache = cls()
            cache.legacy_cache = past_key_values
            return cache

        def get_seq_length(self):
            return 0

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:  # pragma: no cover - compatibility fallback for older Transformers
    class GradientCheckpointingLayer(nn.Module):  # type: ignore[no-redef]
        gradient_checkpointing = False

        def __init__(self, *args, **kwargs):
            super().__init__()

try:
    from transformers.utils import auto_docstring, logging
except ImportError:  # pragma: no cover - compatibility fallback
    from transformers.utils import logging  # type: ignore

    def auto_docstring(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(obj):
            return obj

        return _decorator

try:
    from transformers.utils.deprecation import deprecate_kwarg
except ImportError:  # pragma: no cover - compatibility fallback
    def deprecate_kwarg(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

try:
    from transformers.utils.hub import cached_file
except ImportError:  # pragma: no cover - compatibility fallback
    from transformers.utils import cached_file  # type: ignore


logger = logging.get_logger(__name__)


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


def apply_chunking_to_forward(forward_fn, chunk_size: int, chunk_dim: int, *input_tensors) -> torch.Tensor:
    """Local copy of the HF utility to reduce cross-version import fragility."""
    if len(input_tensors) == 0:
        raise ValueError(f"{input_tensors} has to be a tuple/list of tensors")

    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, found shape {input_tensor.shape[chunk_dim]}"
                )
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk size {chunk_size}"
            )
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """Local copy of the HF utility to reduce cross-version import fragility."""
    index = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, index).detach().clone()
    if layer.bias is not None:
        if dim == 1:
            bias = layer.bias.detach().clone()
        else:
            bias = layer.bias[index].detach().clone()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(weight.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(bias.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(
    heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], torch.LongTensor]:
    """Local copy of the HF utility that was removed from newer Transformers."""
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index

logger = logging.get_logger(__name__)




class MegatronBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MegatronBertModel`]. It is used to instantiate a
    MEGATRON_BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MEGATRON_BERT
    [nvidia/megatron-bert-uncased-345m](https://huggingface.co/nvidia/megatron-bert-uncased-345m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 29056):
            Vocabulary size of the MEGATRON_BERT model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`MegatronBertModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`MegatronBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.

    Examples:

    ```python
    >>> from transformers import MegatronBertConfig, MegatronBertModel

    >>> # Initializing a MEGATRON_BERT google-bert/bert-base-uncased style configuration
    >>> configuration = MegatronBertConfig()

    >>> # Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration
    >>> model = MegatronBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "megatron-bert"

    def __init__(
        self,
        vocab_size=29056,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache



class MegatronBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        # In Megatron, layer-norm is applied after the 1st dropout.
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Megatron BERT moves that layer norm after the drop-out (and to each layer).
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->MegatronBert

class MegatronBertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, layer_idx=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.layer_idx = layer_idx

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = self.query(hidden_states)
        query_layer = query_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(
            1, 2
        )

        is_updated = False
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_layer = self.key(current_states)
            key_layer = key_layer.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(
                1, 2
            )
            value_layer = self.value(current_states)
            value_layer = value_layer.view(
                batch_size, -1, self.num_attention_heads, self.attention_head_size
            ).transpose(1, 2)

            if past_key_values is not None:
                # save all key/value_layer to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if past_key_values is not None:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MegatronBertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs


# Based transformers.models.bert.modeling_bert.BertSelfOutput. Moved LayerNorm to MegatronBertAttention below.

class MegatronBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


# Based transformers.models.bert.modeling_bert.BertAttention. Added LayerNorm.

class MegatronBertAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self = MegatronBertSelfAttention(config, layer_idx=layer_idx)
        self.output = MegatronBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        ln_outputs = self.ln(hidden_states)
        self_outputs = self.self(
            ln_outputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->MegatronBert

class MegatronBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Based on transformers.models.bert.modeling_bert.BertOutput. Moved LayerNorm to MegatronBertLayer below.

class MegatronBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return input_tensor + hidden_states


# Based on transformers.models.bert.modeling_bert.BertLayer. Added LayerNorm.

class MegatronBertLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MegatronBertAttention(config, layer_idx=layer_idx)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = MegatronBertAttention(config, layer_idx=layer_idx)
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = MegatronBertIntermediate(config)
        self.output = MegatronBertOutput(config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return (layer_output,) + outputs

    def feed_forward_chunk(self, attention_output):
        ln_output = self.ln(attention_output)
        intermediate_output = self.intermediate(ln_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class MegatronBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MegatronBertLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        # The final layer norm. We removed the 1st LN, moved LN to each hidden layer and this one
        # is simply the final LN (Transformer's BERT has it attached to each hidden layer).
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
        if use_cache and isinstance(past_key_values, tuple):
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. "
                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
            )
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values,
                output_attentions,
                cache_position,
            )

            # Because we moved the layer-norm at the end of the hidden layer, we have non-normali-
            # zed data here. If that's really needed, we must apply LN to match Transformer's BERT.

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Finalize the hidden states.
        hidden_states = self.ln(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->MegatronBert

class MegatronBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->MegatronBert

class MegatronBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->MegatronBert

class MegatronBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MegatronBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->MegatronBert

class MegatronBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MegatronBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

@auto_docstring
class MegatronBertPreTrainedModel(PreTrainedModel):
    config: MegatronBertConfig
    load_tf_weights = load_tf_weights_in_megatron_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MegatronBertLMPredictionHead):
            module.bias.data.zero_()

@auto_docstring
class MegatronBertModel(MegatronBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://huggingface.co/papers/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = MegatronBertEmbeddings(config)
        self.encoder = MegatronBertEncoder(config)

        self.pooler = MegatronBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = (
                past_key_values[0][0].shape[-2]
                if not isinstance(past_key_values, Cache)
                else past_key_values.get_seq_length()
            )

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@auto_docstring(
    custom_intro="""
    MegatronBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """
)

@auto_docstring
class MegatronBertForMaskedLM(MegatronBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `MegatronBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.cls = MegatronBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


# Previous codes

class ProkBertConfig(MegatronBertConfig):
    model_type = "prokbert"

    def __init__(
        self,
        kmer: int = 6,
        shift: int = 1,
        num_class_labels: int = 2,
        classification_dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kmer = kmer
        self.shift = shift
        self.num_class_labels = num_class_labels
        self.classification_dropout_rate = classification_dropout_rate

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
        self.bert_base_model = bert_base_model
        self.curriculum_hidden_size = curriculum_hidden_size
        self.classification_dropout_rate = classification_dropout_rate

class ProkBertClassificationConfig(ProkBertConfig):
    model_type = "prokbert"
    def __init__(
        self,
        num_labels: int = 2,
        classification_dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.classification_dropout_rate = classification_dropout_rate



class BertForBinaryClassificationWithPooling(nn.Module):
    """
    ProkBERT model for binary classification with custom pooling.

    This model extends a pre-trained `MegatronBertModel` by adding a weighting layer
    to compute a weighted sum over the sequence outputs, followed by a classifier.

    Attributes:
        base_model (MegatronBertModel): The base BERT model.
        weighting_layer (nn.Linear): Linear layer to compute weights for each token.
        dropout (nn.Dropout): Dropout layer.
        classifier (nn.Linear): Linear layer for classification.
    """
    def __init__(self, base_model: MegatronBertModel):
        """
        Initialize the BertForBinaryClassificationWithPooling model.

        Args:
            base_model (MegatronBertModel): A pre-trained `MegatronBertModel` instance.
        """

        super(BertForBinaryClassificationWithPooling, self).__init__()
        self.base_model = base_model
        self.base_model_config_dict = base_model.config.to_dict()
        self.hidden_size = self.base_model_config_dict['hidden_size']
        self.dropout_rate = self.base_model_config_dict['hidden_dropout_prob']

        self.weighting_layer = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, output_pooled_output=False):
        # Modified call to base model to include output_hidden_states
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
        sequence_output = outputs[0]

        # Compute weights for each position in the sequence
        weights = self.weighting_layer(sequence_output)
        weights = torch.nn.functional.softmax(weights, dim=1)

        # Compute weighted sum
        pooled_output = torch.sum(weights * sequence_output, dim=1)

        # Classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Prepare the output as a dictionary
        output = {"logits": logits}

        # Include hidden states in output if requested
        if output_hidden_states:
            output["hidden_states"] = outputs.hidden_states
        if output_pooled_output:
            output["pooled_output"] = pooled_output

        # If labels are provided, compute the loss
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            output["loss"] = loss

        return output

    def save_pretrained(self, save_directory):
        """
        Save the model weights and configuration in a directory.

        Args:
            save_directory (str): Directory where the model and configuration can be saved.
        """
        print('The save pretrained is called!')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        print(f'The save directory is: {save_directory}')
        self.base_model.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load the model weights and configuration from a local directory or Hugging Face Hub.

        Args:
            pretrained_model_name_or_path (str): Directory path where the model and configuration were saved, or name of the model in Hugging Face Hub.

        Returns:
            model: Instance of BertForBinaryClassificationWithPooling.
        """
        # Determine if the path is local or from Hugging Face Hub
        if os.path.exists(pretrained_model_name_or_path):
            # Path is local
            if 'config' in kwargs:
                print('Config is in the parameters')
                config = kwargs['config']

            else:
                config = MegatronBertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            base_model = MegatronBertModel(config=config)
            model = cls(base_model=base_model)
            model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            # Path is from Hugging Face Hub
            config = kwargs.pop('config', None)
            if config is None:
                config = MegatronBertConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

            base_model = MegatronBertModel(config=config)
            model = cls(base_model=base_model)
            model_file = cached_file(pretrained_model_name_or_path, "pytorch_model.bin")
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'), weights_only=True))

        return model


class ProkBertPreTrainedModel(MegatronBertPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProkBertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class ProkBertModel(MegatronBertModel):
    config_class = ProkBertConfig

    def __init__(self, config: ProkBertConfig, **kwargs):
        if not isinstance(config, ProkBertConfig):
            raise ValueError(f"Expected `ProkBertConfig`, got {config.__class__.__module__}.{config.__class__.__name__}")

        super().__init__(config, **kwargs)
        self.config = config
        # One should check if it is a prper prokbert config, if not crafting one.


class ProkBertForMaskedLM(MegatronBertForMaskedLM):
    config_class = ProkBertConfig

    def __init__(self, config: ProkBertConfig, **kwargs):
        if not isinstance(config, ProkBertConfig):
            raise ValueError(f"Expected `ProkBertConfig`, got {config.__class__.__module__}.{config.__class__.__name__}")

        super().__init__(config, **kwargs)
        self.config = config
        # One should check if it is a prper prokbert config, if not crafting one.


class ProkBertForSequenceClassification(ProkBertPreTrainedModel):
    config_class = ProkBertConfig
    base_model_prefix = "bert"

    def __init__(self, config):

        super().__init__(config)
        self.config = config
        self.bert = ProkBertModel(config)
        self.weighting_layer = nn.Linear(self.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.config.classification_dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_class_labels)
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
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_class_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_class_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]

            # Compute weights for each position in the sequence
            weights = self.weighting_layer(sequence_output)
            weights = torch.nn.functional.softmax(weights, dim=1)
            # Compute weighted sum
            pooled_output = torch.sum(weights * sequence_output, dim=1)
            # Classification head
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            loss = None
            if labels is not None:
                loss = self.loss_fct(logits.view(-1, self.config.num_class_labels), labels.view(-1))

            classification_output = SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            return classification_output

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
        except Exception:
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
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = ProkBertModel(config)

        # A weighting layer for pooling the sequence output
        self.weighting_layer = nn.Linear(self.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.config.classification_dropout_rate)

        if config.curriculum_hidden_size != -1:
            self.linear = nn.Linear(self.config.hidden_size, config.curriculum_hidden_size)

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

    def _init_weights(self, module: nn.Module):
        # first let the base class init everything else
        super()._init_weights(module)

        # then catch our pooling head and zero it
        if module is getattr(self, "weighting_layer", None):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        if module is getattr(self, "linear", None):
            initialize_linear_kaiming(module)

        if module is getattr(self, "curricular_face", None):
            nn.init.kaiming_uniform_(module.kernel, a=math.sqrt(self.config.curricular_num_labels))


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
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)

        # Pool the sequence output using a learned weighting (attention-like)
        weights = self.weighting_layer(sequence_output)  # (batch_size, seq_length, 1)
        # Ensure mask shape matches
        if attention_mask.dim() == 2:
            mask = attention_mask
        elif attention_mask.dim() == 4:
            mask = attention_mask.squeeze(1).squeeze(1)  # (batch_size, seq_length)
        else:
            raise ValueError(f"Unexpected attention_mask shape {attention_mask.shape}")

        # Apply mask (masked positions -> -inf before softmax)
        weights = weights.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))

        # Normalize
        weights = torch.nn.functional.softmax(weights, dim=1)  # (batch_size, seq_length)

        # Weighted pooling
        #weights = weights.unsqueeze(-1)                        # (batch_size, seq_length, 1)
        pooled_output = torch.sum(weights * sequence_output, dim=1)  # (batch_size, hidden_size)
        # Classifier head
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class ProkBertForSequenceClassificationExt(ProkBertPreTrainedModel):
    """
    Extensions vs. baseline ProkBertForSequenceClassification:
      - Fixes attention-pooling bug by masking PAD positions using attention_mask
      - Neutral pooling init: weighting_layer starts at zero => uniform pooling over non-masked tokens
      - LN + MLP head on pooled embedding
      - Temperature-controlled attention pooling with learnable temperature (scalar)
    """
    config_class = ProkBertConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = ProkBertModel(config)

        # Attention pooling (token-wise scalar score)
        self.weighting_layer = nn.Linear(self.config.hidden_size, 1)

        # Learnable temperature for pooling: temperature = exp(log_temperature), clamped
        self.log_temperature = nn.Parameter(torch.zeros(()))  # scalar, starts at 0 => temperature=1
        self.temperature_min = float(getattr(config, "pool_temperature_min", 0.1))
        self.temperature_max = float(getattr(config, "pool_temperature_max", 10.0))

        # MLP head on pooled embedding
        eps = float(getattr(config, "layer_norm_eps", 1e-12))
        drop_p = float(getattr(config, "classification_dropout_rate", 0.1))
        hidden_size = int(self.config.hidden_size)
        mlp_hidden = int(getattr(config, "classifier_mlp_hidden_size", max(1, hidden_size // 2)))

        self.mlp_ln = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp_dropout = nn.Dropout(drop_p)
        self.mlp_fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(mlp_hidden, int(self.config.num_class_labels))

        # Loss
        if int(self.config.num_class_labels) == 1:
            self.loss_fct = nn.MSELoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.post_init()

        # --- Custom init for "neutral" pooling + slightly conservative output layer ---
        self._init_ext_head()

    def _init_ext_head(self):
        # Make pooling start neutral: scores = 0 => uniform softmax over non-masked tokens
        with torch.no_grad():
            nn.init.zeros_(self.weighting_layer.weight)
            nn.init.zeros_(self.weighting_layer.bias)

        # Optional: make final classifier layer a bit smaller (reduces early overconfidence)
        init_range = float(getattr(self.config, "initializer_range", 0.02))
        with torch.no_grad():
            nn.init.normal_(self.mlp_fc2.weight, mean=0.0, std=init_range * 0.1)
            nn.init.zeros_(self.mlp_fc2.bias)

    def _get_temperature(self, device: torch.device) -> torch.Tensor:
        # Keep temperature positive and within a reasonable range
        t = torch.exp(self.log_temperature.to(device=device))
        return torch.clamp(t, min=self.temperature_min, max=self.temperature_max)

    @staticmethod
    def _normalize_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert attention_mask to shape (B, L) boolean mask where True means "keep token".
        Handles common shapes: (B, L), (B, 1, 1, L), (B, 1, L).
        """
        if attention_mask is None:
            return None

        mask = attention_mask
        # Common HF forms
        if mask.dim() == 4:
            # (B, 1, 1, L) -> (B, L)
            mask = mask.squeeze(1).squeeze(1)
        elif mask.dim() == 3:
            # (B, 1, L) -> (B, L)
            mask = mask.squeeze(1)

        # Convert to bool: treat >0 as keep
        mask = mask > 0
        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # (B, L, H)

        # --- Temperature-controlled attention pooling with PAD-masking ---
        scores = self.weighting_layer(sequence_output)  # (B, L, 1)

        # Apply temperature (smooth if temperature > 1, sharper if < 1)
        temperature = self._get_temperature(device=scores.device)
        scores = scores / temperature

        # Mask out padding tokens (pooling bug fix)
        keep_mask = self._normalize_attention_mask(attention_mask)  # (B, L) bool or None
        if keep_mask is not None:
            # Guard: if an example is fully masked (shouldn't happen), keep first token to avoid NaNs
            if (keep_mask.sum(dim=1) == 0).any():
                keep_mask = keep_mask.clone()
                keep_mask[(keep_mask.sum(dim=1) == 0), 0] = True

            scores = scores.masked_fill(~keep_mask.unsqueeze(-1), float("-inf"))

        # Softmax in fp32 for stability, then cast back
        weights = torch.softmax(scores.float(), dim=1).to(dtype=sequence_output.dtype)  # (B, L, 1)

        pooled_output = torch.sum(weights * sequence_output, dim=1)  # (B, H)

        # --- LN + MLP head ---
        x = self.mlp_ln(pooled_output)
        x = self.mlp_dropout(x)
        x = self.mlp_fc1(x)
        x = self.mlp_act(x)
        x = self.mlp_dropout(x)
        logits = self.mlp_fc2(x)

        loss = None
        if labels is not None:
            if int(self.config.num_class_labels) == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                loss = self.loss_fct(logits.view(-1, int(self.config.num_class_labels)), labels.view(-1))

        if not return_dict:
            # outputs: (last_hidden_state, pooled_output, hidden_states, attentions) in most BERT-like models
            out = (logits,) + outputs[2:]
            return ((loss,) + out) if loss is not None else out

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
