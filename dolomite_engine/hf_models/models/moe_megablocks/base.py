from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

from ...enums import AttentionHeadType, PositionEmbeddingType
from ...modeling_utils import get_normalization_function
from ..gpt_megatron import GPTMegatronModel, GPTMegatronPreTrainedModel
from .config import MoEMegablocksConfig
from .layer import SparseMoEBlock


class MoEMegablocksPreTrainedModel(GPTMegatronPreTrainedModel):
    config_class = MoEMegablocksConfig
    _no_split_modules = ["SparseMoEBlock"]

    def __init__(self, config: MoEMegablocksConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        assert not self._use_padding_free_transformer, "padding_free input layout doesn't work for MoE layer currently"

    def get_moe_loss(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        cu_seqlens: torch.Tensor,
        router_logits: torch.Tensor,
        num_experts: int,
        num_experts_per_token: int,
        router_aux_loss_coef: float,
        output_router_logits: bool,
    ) -> torch.Tensor:
        loss = self.get_autoregressive_language_modeling_loss(lm_logits, labels, cu_seqlens)

        load_balancing_loss = None
        if output_router_logits:
            load_balancing_loss = load_balancing_loss_func(router_logits, num_experts, num_experts_per_token)
            if loss is not None:
                loss += router_aux_loss_coef * load_balancing_loss

        return loss, load_balancing_loss


class MoEMegablocksModel(MoEMegablocksPreTrainedModel, GPTMegatronModel):
    def __init__(self, config: MoEMegablocksConfig, **kwargs) -> None:
        MoEMegablocksPreTrainedModel.__init__(self, config, **kwargs)

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        assert (
            self.embed_dim % self.num_heads == 0
        ), f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.embed_dim // self.num_heads

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                SparseMoEBlock(
                    config,
                    self.normalization_implementation,
                    self.attention_implementation,
                    self._use_padding_free_transformer,
                    layer_idx=i,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = get_normalization_function(
            config.normalization_function,
            self.embed_dim,
            eps=config.layer_norm_epsilon,
            normalization_implementation=self.normalization_implementation,
        )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self._setup_positional_encoding()

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor = None,
        past_key_values: List[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
        output_router_logits: bool = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        (
            output_attentions,
            output_hidden_states,
            use_cache,
            return_dict,
            input_shape,
            hidden_states,
            attention_mask,
            position_ids,
            alibi_bias,
            rope_cos_sin,
            past_key_values,
            output_router_logits,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_router_logits=output_router_logits,
        )

        # ==========================================================================================
        # padding_free:
        #     attention_mask -> None
        # flash:
        #     attention_mask -> (batch_size, key_length)
        # else:
        #     attention_mask -> (batch_size, 1, query_length, key_length)
        # ==========================================================================================

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = [] if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                outputs = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    position_ids,
                    alibi_bias,
                    rope_cos_sin,
                    use_cache,
                    output_attentions,
                    cu_seqlens,
                    max_seqlen,
                    output_router_logits,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    alibi_bias=alibi_bias,
                    rope_cos_sin=rope_cos_sin,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    output_router_logits=output_router_logits,
                )

            hidden_states = outputs[0]
            if use_cache:
                presents.append(outputs[1])

            if output_attentions:
                all_self_attentions += (outputs[2 if use_cache else 1],)

            if output_router_logits:
                all_router_logits += (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_router_logits]
                if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            router_logits=all_router_logits,
        )

    def _prepare_a_bunch_of_stuff(
        self,
        input_ids: torch.Tensor = None,
        past_key_values: List[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
        output_router_logits: bool = False,
    ) -> Tuple[
        bool,
        bool,
        bool,
        bool,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Union[Tuple[torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        return super()._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        ) + (output_router_logits,)
