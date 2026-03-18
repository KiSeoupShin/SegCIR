import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.utils.checkpoint
from transformers import BertModel, BertConfig, BertLMHeadModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertAttention, BertIntermediate, BertOutput
# from lavis.models.blip2_models.Qformer import BertModel, BertConfig, BertLMHeadModel, BertEmbeddings, BertPooler, BertAttention, BertIntermediate, BertOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.activations import ACT2FN


### BLIP_CUSTOM
# class BertPredictionHeadTransform(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         if isinstance(config.hidden_act, str):
#             self.transform_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.transform_act_fn = config.hidden_act
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.transform_act_fn(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states

# class BertLMPredictionHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.transform = BertPredictionHeadTransform(config)

#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         self.bias = nn.Parameter(torch.zeros(config.vocab_size))

#         # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
#         self.decoder.bias = self.bias

#     def _tie_weights(self):
#         self.decoder.bias = self.bias

#     def forward(self, hidden_states):
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.decoder(hidden_states)
#         return hidden_states

# class BertOnlyMLMHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.predictions = BertLMPredictionHead(config)

#     def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
#         prediction_scores = self.predictions(sequence_output)
#         return prediction_scores

# class CrossAttentionOnlyBertLayer(nn.Module):
#     def __init__(self, config, layer_num=0):
#         super().__init__()
#         self.config = config
#         self.chunk_size_feed_forward = config.chunk_size_feed_forward
#         self.seq_len_dim = 1
#         self.layer_num = layer_num
        
#         # Disable self-attention - we'll just pass hidden_states directly
        
#         # Make sure the model is configured as a decoder with cross-attention
#         self.is_decoder = True
#         self.add_cross_attention = True
        
#         # We only use the cross-attention module from BertAttention
#         self.crossattention = BertAttention(config, is_cross_attention=True)
#         self.has_cross_attention = True
        
#         self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)
        
#         # Add separate FFNs for query part, matching LAVIS architecture
#         self.intermediate_query = BertIntermediate(config)
#         self.output_query = BertOutput(config)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_value=None,
#         output_attentions=False,
#         query_length=0,
#         gamma=None,
#         beta=None,
#     ):
#         # Skip self-attention completely - use hidden_states directly
#         attention_output = hidden_states
#         outputs = () # For holding attention outputs if needed
        
#         # For compatibility with BertLayer interface, creating placeholder for self-attention cache
#         # Since we don't use self-attention, we just create empty placeholders
#         self_attention_cache = ((None, None),) if past_key_value is not None else None
#         present_key_value = self_attention_cache
        
#         # Handle query_length portion separately
#         if query_length > 0:
#             query_attention_output = attention_output[:, :query_length, :]
            
#             # Do cross-attention only on the query portion
#             if encoder_hidden_states is not None:
#                 assert (
#                     encoder_hidden_states is not None
#                 ), "encoder_hidden_states must be given for cross-attention layers"
                
#                 # Extract appropriate past key value for cross-attention
#                 cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
                
#                 cross_attention_outputs = self.crossattention(
#                     query_attention_output,
#                     attention_mask,
#                     head_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     past_key_value=cross_attn_past_key_value,
#                     output_attentions=output_attentions,
#                 )
#                 query_attention_output = cross_attention_outputs[0]

#                 if gamma is not None and beta is not None:
#                     query_attention_output = gamma * attention_output + beta
                
#                 # Add cross attentions if we output attention weights
#                 outputs = outputs + cross_attention_outputs[1:-1]
                
#                 # Update present_key_value with cross-attention cache
#                 if cross_attention_outputs[-1] is not None:
#                     present_key_value = cross_attention_outputs[-1]
            
#             # Apply feed-forward to query part using query-specific FFN
#             query_layer_output = apply_chunking_to_forward(
#                 self.feed_forward_chunk_query,
#                 self.chunk_size_feed_forward,
#                 self.seq_len_dim,
#                 query_attention_output,
#             )
            
#             # Process non-query part if it exists
#             if attention_output.shape[1] > query_length:
#                 non_query_output = attention_output[:, query_length:, :]
                
#                 # For non-query part, we just apply feed-forward without cross-attention
#                 non_query_layer_output = apply_chunking_to_forward(
#                     self.feed_forward_chunk,
#                     self.chunk_size_feed_forward,
#                     self.seq_len_dim,
#                     non_query_output,
#                 )
                
#                 # Combine query and non-query outputs
#                 layer_output = torch.cat([query_layer_output, non_query_layer_output], dim=1)
#             else:
#                 # If there's only query portion
#                 layer_output = query_layer_output
#         else:
#             # No query length specified - apply cross-attention to all
#             if encoder_hidden_states is not None:
#                 # Extract appropriate past key value for cross-attention
#                 cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
                
#                 cross_attention_outputs = self.crossattention(
#                     attention_output,
#                     attention_mask,
#                     head_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     past_key_value=cross_attn_past_key_value,
#                     output_attentions=output_attentions,
#                 )
#                 attention_output = cross_attention_outputs[0]

#                 if gamma is not None and beta is not None:
#                     attention_output = gamma * attention_output + beta
                
#                 # Add cross attentions if we output attention weights
#                 outputs = outputs + cross_attention_outputs[1:-1]
                
#                 # Update present_key_value with cross-attention cache
#                 if cross_attention_outputs[-1] is not None:
#                     present_key_value = cross_attention_outputs[-1]
            
#             # Apply feed-forward to all
#             layer_output = apply_chunking_to_forward(
#                 self.feed_forward_chunk,
#                 self.chunk_size_feed_forward,
#                 self.seq_len_dim,
#                 attention_output,
#             )
        
#         # Finalize outputs
#         outputs = (layer_output,) + outputs
#         outputs = outputs + (present_key_value,)
        
#         return outputs

#     def feed_forward_chunk(self, attention_output):
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)
#         return layer_output

#     def feed_forward_chunk_query(self, attention_output):
#         intermediate_output = self.intermediate_query(attention_output)
#         layer_output = self.output_query(intermediate_output, attention_output)
#         return layer_output


# class CrossAttentionOnlyBertEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         # Use our custom layer with only cross-attention, passing layer index
#         self.layer = nn.ModuleList(
#             [CrossAttentionOnlyBertLayer(config, i) for i in range(config.num_hidden_layers)]
#         )

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=False,
#         output_hidden_states=False,
#         return_dict=True,
#         query_length=0,
#         gamma_list=None,
#         beta_list=None,
#     ):
#         # Implementation similar to LAVIS BertEncoder, but with our custom layer
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attentions = () if output_attentions else None
#         all_cross_attentions = (
#             () if output_attentions and self.config.add_cross_attention else None
#         )

#         next_decoder_cache = () if use_cache else None
        
#         for i in range(self.config.num_hidden_layers):
#             layer_module = self.layer[i]
            
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             layer_head_mask = head_mask[i] if head_mask is not None else None
#             past_key_value = past_key_values[i] if past_key_values is not None else None

#             gamma = gamma_list[i] if gamma_list is not None else None
#             beta = beta_list[i] if beta_list is not None else None

#             if getattr(self.config, "gradient_checkpointing", False) and self.training:
#                 if use_cache:
#                     logger.warn(
#                         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                     )
#                     use_cache = False

#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         return module(
#                             *inputs, past_key_value, output_attentions, query_length
#                         )

#                     return custom_forward

#                 layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(layer_module),
#                     hidden_states,
#                     attention_mask,
#                     layer_head_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                 )
#             else:
#                 layer_outputs = layer_module(
#                     hidden_states,
#                     attention_mask,
#                     layer_head_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     past_key_value,
#                     output_attentions,
#                     query_length,
#                     gamma,
#                     beta,
#                 )

#             hidden_states = layer_outputs[0]
            
#             if use_cache:
#                 next_decoder_cache += (layer_outputs[-1],)
                
#             if output_attentions:
#                 # 우리 모델은 셀프어텐션이 없으므로 None 추가
#                 all_self_attentions = all_self_attentions + (None,)
                
#                 # 크로스 어텐션 결과 저장 (존재하는 경우)
#                 if self.config.add_cross_attention and len(layer_outputs) > 2:
#                     all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [
#                     hidden_states,
#                     next_decoder_cache,
#                     all_hidden_states,
#                     all_self_attentions,
#                     all_cross_attentions,
#                 ]
#                 if v is not None
#             )
            
#         return BaseModelOutputWithPastAndCrossAttentions(
#             last_hidden_state=hidden_states,
#             past_key_values=next_decoder_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#             cross_attentions=all_cross_attentions,
#         )


# class CrossAttentionOnlyBertModel(BertModel):
#     def __init__(self, config, add_pooling_layer=True):
#         super(BertModel, self).__init__(config)
#         self.config = config
        
#         # Set the model as decoder with cross-attention
#         config.is_decoder = True
#         config.add_cross_attention = True

#         self.embeddings = BertEmbeddings(config)
#         # Use our custom encoder with only cross-attention
#         self.encoder = CrossAttentionOnlyBertEncoder(config)

#         self.pooler = BertPooler(config) if add_pooling_layer else None

#         # Add the missing attribute
#         self.attn_implementation = config._attn_implementation if hasattr(config, '_attn_implementation') else 'eager'
#         self.position_embedding_type = config.position_embedding_type if hasattr(config, 'position_embedding_type') else 'absolute'

#         # Initialize weights and apply final processing
#         self.post_init()
    
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         head_mask=None,
#         query_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         is_decoder=False,
#         gamma_list=None,
#         beta_list=None,
#     ):
#         r"""
#         encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
#             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
#             the model is configured as a decoder.
#         encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
#             the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.
#         past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
#             Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
#             If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
#             (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
#             instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
#         use_cache (:obj:`bool`, `optional`):
#             If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
#             decoding (see :obj:`past_key_values`).
#         """
#         output_attentions = (
#             output_attentions
#             if output_attentions is not None
#             else self.config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states
#             if output_hidden_states is not None
#             else self.config.output_hidden_states
#         )
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )

#         # use_cache = use_cache if use_cache is not None else self.config.use_cache

#         if input_ids is None:
#             assert (
#                 query_embeds is not None
#             ), "You have to specify query_embeds when input_ids is None"

#         # past_key_values_length
#         past_key_values_length = (
#             past_key_values[0][0].shape[2] - self.config.query_length
#             if past_key_values is not None
#             else 0
#         )

#         query_length = query_embeds.shape[1] if query_embeds is not None else 0

#         embedding_output = self.embeddings(
#             input_ids=input_ids,
#             position_ids=position_ids,
#             query_embeds=query_embeds,
#             past_key_values_length=past_key_values_length,
#         )

#         input_shape = embedding_output.size()[:-1]
#         batch_size, seq_length = input_shape
#         device = embedding_output.device

#         if attention_mask is None:
#             attention_mask = torch.ones(
#                 ((batch_size, seq_length + past_key_values_length)), device=device
#             )

#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         if is_decoder:
#             extended_attention_mask = self.get_extended_attention_mask(
#                 attention_mask,
#                 input_ids.shape,
#                 device,
#                 is_decoder,
#                 has_query=(query_embeds is not None),
#             )
#         else:
#             extended_attention_mask = self.get_extended_attention_mask(
#                 attention_mask, input_shape, device, is_decoder
#             )

#         # If a 2D or 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         if encoder_hidden_states is not None:
#             if type(encoder_hidden_states) == list:
#                 encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
#                     0
#                 ].size()
#             else:
#                 (
#                     encoder_batch_size,
#                     encoder_sequence_length,
#                     _,
#                 ) = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

#             if type(encoder_attention_mask) == list:
#                 encoder_extended_attention_mask = [
#                     self.invert_attention_mask(mask) for mask in encoder_attention_mask
#                 ]
#             elif encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#                 encoder_extended_attention_mask = self.invert_attention_mask(
#                     encoder_attention_mask
#                 )
#             else:
#                 encoder_extended_attention_mask = self.invert_attention_mask(
#                     encoder_attention_mask
#                 )
#         else:
#             encoder_extended_attention_mask = None

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         encoder_outputs = self.encoder(
#             embedding_output,
#             attention_mask=extended_attention_mask,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_extended_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             query_length=query_length,
#             gamma_list=gamma_list,
#             beta_list=beta_list,
#         )
#         sequence_output = encoder_outputs[0]
#         pooled_output = (
#             self.pooler(sequence_output) if self.pooler is not None else None
#         )

#         if not return_dict:
#             return (sequence_output, pooled_output) + encoder_outputs[1:]

#         return BaseModelOutputWithPoolingAndCrossAttentions(
#             last_hidden_state=sequence_output,
#             pooler_output=pooled_output,
#             past_key_values=encoder_outputs.past_key_values,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#             cross_attentions=encoder_outputs.cross_attentions,
#         )


# class CrossAttentionOnlyBertLMHeadModel(BertLMHeadModel):
#     def __init__(self, config):
#         super(BertLMHeadModel, self).__init__(config)
        
#         # Set the model as decoder with cross-attention
#         config.is_decoder = True
#         config.add_cross_attention = True
        
#         # Use our custom model with only cross-attention
#         self.bert = CrossAttentionOnlyBertModel(config, add_pooling_layer=False)
#         self.cls = BertOnlyMLMHead(config)
        
#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         encoder_hidden_states=None,  # This is required for cross-attention
#         encoder_attention_mask=None,  # This is required for cross-attention
#         labels=None,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         **kwargs
#     ):
#         # Makes sure encoder_hidden_states is provided
#         if encoder_hidden_states is None:
#             raise ValueError(
#                 "CrossAttentionOnlyBertLMHeadModel requires encoder_hidden_states for cross-attention."
#             )
            
#         return super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             labels=labels,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             **kwargs
#         )


### FILM_CUSTOM
class BertPredictionHeadTransform(nn.Module):
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

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

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

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class CrossAttentionOnlyBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.is_decoder = True
        self.add_cross_attention = True

        self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        gamma=None,
        beta=None,
    ):
        attention_output = hidden_states
        cross_attn_present_key_value = None
        cross_attention_outputs = None

        if encoder_hidden_states is not None:
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]

            # FiLM 연산 적용
            if gamma is not None and beta is not None:
                attention_output = gamma * attention_output + beta

            if cross_attention_outputs[-1] is not None:
                cross_attn_present_key_value = cross_attention_outputs[-1]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,)
        if output_attentions and cross_attention_outputs is not None:
            outputs += (None, cross_attention_outputs[1])

        outputs = outputs + (cross_attn_present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CrossAttentionOnlyBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([CrossAttentionOnlyBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        gamma_list=None,
        beta_list=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            gamma = gamma_list[i] if gamma_list is not None else None
            beta = beta_list[i] if beta_list is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                gamma=gamma,
                beta=beta,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (None,)
                if self.config.add_cross_attention and len(layer_outputs) > 2:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class CrossAttentionOnlyBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.config = config

        config.is_decoder = True
        config.add_cross_attention = True

        self.embeddings = BertEmbeddings(config)
        self.encoder = CrossAttentionOnlyBertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = getattr(config, '_attn_implementation', 'eager')
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        gamma_list=None,
        beta_list=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=self.embeddings.word_embeddings.weight.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.embeddings.word_embeddings.weight.device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values[0][0].size(2) if past_key_values is not None else 0,
        )

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gamma_list=gamma_list,
            beta_list=beta_list,
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


class CrossAttentionOnlyBertLMHeadModel(BertLMHeadModel):
    def __init__(self, config):
        super(BertLMHeadModel, self).__init__(config)
        
        # Set the model as decoder with cross-attention
        config.is_decoder = True
        config.add_cross_attention = True
        
        # Use our custom model with only cross-attention
        self.bert = CrossAttentionOnlyBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,  # This is required for cross-attention
        encoder_attention_mask=None,  # This is required for cross-attention
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # Makes sure encoder_hidden_states is provided
        if encoder_hidden_states is None:
            raise ValueError(
                "CrossAttentionOnlyBertLMHeadModel requires encoder_hidden_states for cross-attention."
            )
            
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )