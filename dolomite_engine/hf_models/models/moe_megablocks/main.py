from typing import List, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from ..gpt_megatron import GPTMegatronForCausalLM
from .base import MoEMegablocksModel, MoEMegablocksPreTrainedModel
from .config import MoEMegablocksConfig


class MoEMegablocksForCausalLM(MoEMegablocksPreTrainedModel, GPTMegatronForCausalLM):
    def __init__(self, config: MoEMegablocksConfig, **kwargs) -> None:
        MoEMegablocksPreTrainedModel.__init__(self, config, **kwargs)

        self.transformer = MoEMegablocksModel(config, **kwargs)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]] = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: Union[torch.Tensor, List[List[int]]] = None,
        position_ids: Union[torch.Tensor, List[List[int]]] = None,
        inputs_embeds: Union[torch.Tensor, List[List[float]]] = None,
        labels: Union[torch.Tensor, List[List[int]]] = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
        output_router_logits: bool = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        if self._use_padding_free_transformer and output_router_logits:
            raise NotImplementedError("padding_free is not supported with load_balancing_loss_func currently")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        (
            input_ids,
            position_ids,
            token_type_ids,
            labels,
            cu_seqlens,
            max_seqlen,
        ) = self.prepare_inputs_for_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # ==========================================================================================
        # padding_free:
        #     input_ids -> (total_q)
        #     attention_mask -> None
        #     position_ids -> (total_q)
        # else:
        #     input_ids -> (batch_size, query_length)
        #     attention_mask -> None or (batch_size, key_length)
        #     position_ids -> None or (batch_size, key_length)
        # ==========================================================================================

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_router_logits=output_router_logits,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        loss, load_balancing_loss = self.get_moe_loss(
            lm_logits=lm_logits,
            labels=labels,
            cu_seqlens=cu_seqlens,
            router_logits=transformer_outputs.router_logits if return_dict else transformer_outputs[-1],
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_tok,
            router_aux_loss_coef=self.router_aux_loss_coef,
            output_router_logits=output_router_logits,
        )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            if output_router_logits:
                output = (load_balancing_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=load_balancing_loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            router_logits=transformer_outputs.router_logits,
        )