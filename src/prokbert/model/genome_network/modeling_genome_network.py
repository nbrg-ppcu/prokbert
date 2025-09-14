from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils.import_utils import is_triton_available
from transformers.utils import is_flash_attn_2_available, logging
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from ...models import ProkBertConfig, ProkBertModel
from .configuration_genome_network import GenomeNetworkConfig


if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func


logger = logging.get_logger(__name__)


def eager_attention_forward(
    module: "GenomeNetworkAttention",
    qkv: torch.Tensor,
    attention_mask: torch.FloatTensor,
    sliding_window_attention_mask: torch.FloatTensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor]:

    # 3 x (batch size, n_heads, seq_len, head_dim) <- (batch size, seqlen, 3, nheads, head_dim)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)

    scale = module.head_dim ** -0.5 # sqrt(head_dim)
    # (bs, n_heads, seq_len, seq_len) <- (bs, n_heads, seq_len, head_dim) @ (batch size, n_heads, head_dim, seq_len)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_attention_mask

    if attention_mask is not None:
        # (bs, n_heads, seq_len, seq_len) + (bs, 1, 1, seq_len)
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32 for stability.
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
    sliding_window_attention_mask: torch.Tensor,
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_attention_mask

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

        if layer_id is not None and layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

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

        if layer_id and layer_id == 0:
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
        sliding_window_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:

        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            sliding_window_attention_mask=sliding_window_attention_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_outputs[0] # residual connection

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


class GenomeNetwork(GenomeNetworkPreTrainedModel):
    def __init__(self, config: GenomeNetworkConfig) -> None:
        super().__init__(config)
        self.config = config

        self.layers = nn.ModuleList(
            [GenomeNetworkEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False

        self.pooling = None # ! for now

        self.post_init() # initalize weights and apply final processing

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...] | BaseModelOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        self._maybe_set_compile()

        assert inputs_embeds.dim() == 3, f"Expected input_embeds to be of shape (batch_size / genoms, genes, embedding), but got {inputs_embeds.shape}."
        batch_size, gene_len, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, gene_len), device=device)

        extended_attention_mask, sliding_window_attention_mask = self.update_attention_mask(
            attention_mask, output_attentions
        )

        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    extended_attention_mask,
                    sliding_window_attention_mask=sliding_window_attention_mask,
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    sliding_window_attention_mask=sliding_window_attention_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def update_attention_mask(self, attention_mask: torch.Tensor, output_attentions: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if output_attentions:
            if self.config._attn_implementation == "sdpa":
                logger.warning(
                    "Outputting attentions is only supported with the 'eager' attention implementation, "
                    'not with "sdpa". Falling back to `attn_implementation="eager"`.'
                )
                self.config._attn_implementation = "eager"
            elif self.config._attn_implementation != "eager":
                logger.warning(
                    "Outputting attentions is only supported with the eager attention implementation, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`. '
                    "Setting `output_attentions=False`."
                )
        global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        distance = torch.abs(rows - rows.T)
        window_mask = ((distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attention_mask.device))
        sliding_window_attention_mask = global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min)
        return global_attention_mask, sliding_window_attention_mask


class GenomeNetworkForMaskedLM(GenomeNetworkPreTrainedModel):
    def __init__(
        self,
        config: GenomeNetworkConfig,
        embedding_model: Optional[PreTrainedModel] = None,
        embedding_config: Optional[ProkBertConfig] = None,
    ) -> None:
        if not isinstance(config, GenomeNetworkConfig):
            raise ValueError(
                f"Expected `GenomeNetworkConfig`, got {config.__class__.__module__}.{config.__class__.__name__}"
            )

        super().__init__(config)

        self.config = config

        if embedding_model is None and embedding_config is not None:
            self.embedding_model = ProkBertModel(embedding_config)
        elif embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            raise ValueError(
                "Either `embedding_model` or `embedding_config` has to be provided."
            )

        # freeze embedding model
        for param in self.embedding_model.parameters():
            param.requires_grad = False

        self.bert = GenomeNetwork(config)
        self.post_init()  # initalize weights and apply final processing

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,           # (bs, genes, seq len)
        inputs_embeds: Optional[torch.FloatTensor] = None,      # (bs, genes, hidden size)
        attention_mask: Optional[torch.Tensor] = None,          # (genes, seq_len)
        attention_mask_genome: Optional[torch.Tensor] = None,   # (bs, genes)
        token_type_ids: Optional[torch.LongTensor] = None,      # (genes, seq_len)
        labels: Optional[torch.Tensor] = None,                  # (bs, genes, hidden size)
        labels_mask: Optional[torch.BoolTensor] = None,         # (bs, genes)
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[tuple,  MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self._maybe_set_compile()

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        if input_ids is not None and (attention_mask is None or token_type_ids is None):
            raise ValueError("If input_ids is used, attention_mask and token_type_ids have to be provided.")

        if input_ids is not None:
            embedding_outputs = [
                self.embedding_model(
                    input_ids=input_id,
                    attention_mask=attn_mask,
                    token_type_ids=token_type_id
                ).pooler_output.detach()
                for input_id, attn_mask, token_type_id in zip(input_ids, attention_mask, token_type_ids)
            ]
            inputs_embeds = torch.stack(embedding_outputs, dim=0)

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_genome,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if labels is not None and labels_mask is not None:
            last_hidden_state = last_hidden_state[labels_mask].view(-1, self.config.hidden_size)
            labels_embeds = inputs_embeds[labels_mask].view(-1, self.config.hidden_size)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()  # squared L2 norm
            masked_lm_loss = loss_fct(
                last_hidden_state,
                labels_embeds,
            )

        if not return_dict:
            return ((masked_lm_loss,) + outputs[2:]) if masked_lm_loss is not None else outputs

        return MaskedLMOutput(
            loss=masked_lm_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
