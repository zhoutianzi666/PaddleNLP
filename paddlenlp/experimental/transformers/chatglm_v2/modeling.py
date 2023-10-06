

# Copyright (c) 2023 ChatGLM2-6B Model Team and PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.utils import map_structure
from paddlenlp.experimental.transformers.fused_transformer_layers import (
    FusedMultiTransformer,
)
from paddlenlp_ops import get_padding_offset

from paddlenlp.experimental.transformers.generation_utils import GenerationInferenceModel
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPast,
    ModelOutput,
)
from paddlenlp.transformers import ChatGLMv2Config, ChatGLMv2PretrainedModel
#from paddlenlp.transformers import CHATGLM_V2_PRETRAINED_RESOURCE_FILES_MAP
from paddlenlp.transformers.chatglm_v2.modeling import Embedding,RotaryEmbedding,RMSNorm

from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)

__all__ = [
    "ChatGLMv2ForCausalLMInferenceModel",
]


@register_base_model
class ChatGLMv2InferenceModel(ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config, empty_init=True):
        super().__init__(config)
        self.embedding = Embedding(config)

        # Rotary positional embeddings
        self.max_sequence_length = config.max_sequence_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2)
        
        if config.tensor_parallel_degree > 1:
            if config.tensor_parallel_degree > 2:
                raise ValueError(
                    "ChatGLM2 does not support `tensor_parallel_degree` > 2. Consider using Sharding stage 3"
                )
            self.output_layer = fleet.meta_parallel.ColumnParallelLinear(
                config.hidden_size,
                config.padded_vocab_size,
                has_bias=False,
                gather_output=not config.tensor_parallel_output,
            )
        else:
            self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias_attr=False)


        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_heads
        self.head_size = self.hidden_size // self.num_heads
        self.multi_query_group_num = config.multi_query_group_num

        ln_scale_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.input_layernorm.weight".format(i))
            for i in range(config.num_hidden_layers)
        ]

        qkv_weight_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.qkv_weight".format(i))
            for i in range(config.num_hidden_layers)
        ]
        qkv_bias_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.qkv_bias".format(i)) for i in range(config.num_hidden_layers)
        ]

        out_proj_weight_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.self_attention.dense.weight".format(i))
            for i in range(config.num_hidden_layers)
        ]

        ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.post_attention_layernorm.weight".format(i))
            for i in range(config.num_hidden_layers)
        ]

        ffn1_weight_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.mlp.dense_h_to_4h.weight".format(i))
            for i in range(config.num_hidden_layers)
        ]

        ffn2_weight_attrs = [
            paddle.ParamAttr(name="encoder.layers.{}.mlp.dense_4h_to_h.weight".format(i))
            for i in range(config.num_hidden_layers)
        ]

        self.transformer_block = FusedMultiTransformer(
            config.hidden_size,
            config.num_attention_heads,
            config.ffn_hidden_size,
            dropout_rate=0.0,
            activation="swiglu",
            normalize_before=True,
            num_layers=config.num_hidden_layers,
            nranks=1,
            ring_id=-1,
            ln_scale_attrs=ln_scale_attrs,
            qkv_weight_attrs=qkv_weight_attrs,
            qkv_bias_attrs=qkv_bias_attrs,
            linear_weight_attrs=out_proj_weight_attrs,
            ffn_ln_scale_attrs=ffn_ln_scale_attrs,
            ffn1_weight_attrs=ffn1_weight_attrs,
            ffn2_weight_attrs=ffn2_weight_attrs,
            epsilon=config.layernorm_epsilon,
            norm_type="rmsnorm",
        )

        self.post_layer_norm = config.post_layer_norm
        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else nn.LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, epsilon=config.layernorm_epsilon)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def remove_padding(self, input_ids, seq_lens_this_time):
        cum_offsets_now = paddle.cumsum(paddle.max(seq_lens_this_time) - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(
            input_ids, cum_offsets_now, token_num, seq_lens_this_time
        )
        return ids_remove_padding, padding_offset, cum_offsets

    def forward(
        self,
        input_ids=None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds=None,
        use_cache=None,
        cache_kvs=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        **kwargs,
    ):

        # kwargs["cache"] is used used to distinguish between encoder and decoder phase.
        past_key_values = kwargs.get("cache", None)
        is_decoder = past_key_values is not None

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not is_decoder:
            ids_remove_padding, padding_offset, cum_offsets = self.remove_padding(input_ids, seq_len_encoder)
        else:
            ids_remove_padding = input_ids
            padding_offset = None
            cum_offsets = None

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding.word_embeddings(ids_remove_padding)
        hidden_states = inputs_embeds

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.max_sequence_length)

        #print("position_ids",position_ids)

        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
            rotary_pos_emb = rotary_pos_emb[:,:seq_length,:,:]
            #rotary_pos_emb = rotary_pos_emb[0:1,:seq_length,:,:]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]


        # make it to be [2, 1, seq_len, rotary_dim]
        rotary_pos_emb = rotary_pos_emb.transpose([3, 0, 1, 2])
        #print("rotary_pos_emb.shape", rotary_pos_emb.shape)
        if is_decoder:
            # 这个修改是为了和masked_multihead_attention中位置编码的计算逻辑相匹配，累啊！
            rotary_pos_emb = rotary_pos_emb.unsqueeze(-1).tile([1, 1, 1, 1, 2]).reshape([2,-1,1,64])

        # Run encoder. 
        seq_lens = seq_len_decoder if is_decoder else seq_len_encoder
        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                    input_ids,
                    hidden_states,
                    cum_offsets=cum_offsets,
                    padding_offset=padding_offset,
                    attn_mask=paddle.cast(attention_mask, dtype=hidden_states.dtype),
                    caches=cache_kvs,
                    pre_caches=None,
                    pre_caches_length=0,
                    seq_lens=seq_lens,
                    rotary_embs=paddle.cast(rotary_pos_emb, "float32"),
                    rotary_emb_dims=1,
                    time_step=paddle.increment(paddle.shape(attention_mask)[-1], -1) if is_decoder else None)

        hidden_states = self.final_layernorm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, None, None] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        prefix=""
        self.embedding.word_embeddings.weight.set_value(state_dict.pop(prefix+"embedding.word_embeddings.weight"))
        self.final_layernorm.weight.set_value(state_dict.pop(prefix+"encoder.final_layernorm.weight"))
        self.output_layer.weight.set_value(state_dict.pop(prefix+"output_layer.weight"))

        for i in range(self.num_layers):
            ln_scale = state_dict.pop(prefix+"encoder.layers.{}.input_layernorm.weight".format(i))

            q_weight = state_dict.pop(prefix+"encoder.layers.{}.self_attention.query.weight".format(i))
            k_weight = state_dict.pop(prefix+"encoder.layers.{}.self_attention.key.weight".format(i))
            v_weight = state_dict.pop(prefix+"encoder.layers.{}.self_attention.value.weight".format(i))
            q_bias = state_dict[prefix+"encoder.layers.{}.self_attention.query.bias".format(i)]
            k_bias = state_dict[prefix+"encoder.layers.{}.self_attention.key.bias".format(i)]
            v_bias = state_dict[prefix+"encoder.layers.{}.self_attention.value.bias".format(i)]
            
            import numpy as np
            k_weight = k_weight.reshape([4096, 2, 1, 128])
            k_bias = k_bias.reshape([2, 1, 128])
            v_weight = v_weight.reshape([4096, 2, 1, 128])
            v_bias = v_bias.reshape([2, 1, 128])

            k_weight = np.tile(k_weight, [1, 1, self.num_heads // self.multi_query_group_num, 1])
            v_weight = np.tile(v_weight, [1, 1, self.num_heads // self.multi_query_group_num, 1])
            k_bias = np.tile(k_bias, [1, self.num_heads // self.multi_query_group_num, 1])
            v_bias = np.tile(v_bias, [1, self.num_heads // self.multi_query_group_num, 1])
            k_weight = k_weight.reshape([4096, 4096])
            k_bias = k_bias.reshape([4096])
            v_weight = v_weight.reshape([4096, 4096])
            v_bias = v_bias.reshape([4096])

            concated_qkv_weight = np.concatenate([q_weight, k_weight, v_weight], axis=-1)
            concated_qkv_weight = concated_qkv_weight.transpose(1, 0)
            concated_qkv_weight = concated_qkv_weight.reshape([3 * self.hidden_size , self.hidden_size])
            concated_qkv_weight = paddle.to_tensor(concated_qkv_weight)

            concated_qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=-1)
            concated_qkv_bias = concated_qkv_bias.reshape([3 * self.hidden_size])
            concated_qkv_bias = paddle.to_tensor(concated_qkv_bias)

            out_proj_weight = state_dict.pop(prefix+"encoder.layers.{}.self_attention.dense.weight".format(i))

            ffn_ln_scale = state_dict.pop(prefix+"encoder.layers.{}.post_attention_layernorm.weight".format(i))

            ffn1_weight = state_dict.pop(prefix+"encoder.layers.{}.mlp.dense_h_to_4h.weight".format(i))
            ffn1_weight_0 = ffn1_weight[..., 0::2]
            ffn1_weight_1 = ffn1_weight[..., 1::2]
            ffn1_weight = paddle.concat([ffn1_weight_0, ffn1_weight_1], axis=-1)
            ffn2_weight = state_dict.pop(prefix+"encoder.layers.{}.mlp.dense_4h_to_h.weight".format(i))

            self.transformer_block.ln_scales[i].set_value(ln_scale)

            self.transformer_block.qkv_weights[i].set_value(concated_qkv_weight)
            self.transformer_block.qkv_biases[i].set_value(concated_qkv_bias)

            self.transformer_block.linear_weights[i].set_value(out_proj_weight)

            self.transformer_block.ffn_ln_scales[i].set_value(ffn_ln_scale)

            self.transformer_block.ffn1_weights[i].set_value(ffn1_weight)
            self.transformer_block.ffn2_weights[i].set_value(ffn2_weight)

class ChatGLMv2ForCausalLMInferenceModel(GenerationInferenceModel, ChatGLMv2PretrainedModel):
    def __init__(self, config: ChatGLMv2Config):
        super().__init__(config)
        self.max_sequence_length = config.max_sequence_length
        self.chatglm_v2 = ChatGLMv2InferenceModel(config)

    @classmethod
    def get_cache_kvs_shape(cls, config: ChatGLMv2Config, max_batch_size: int = None, max_length: int = None):
        """get cache_kvs tensor for opt model

        Args:
            max_batch_size (int): the max batch size
            max_length (int | None, optional): the max_length of cache_kvs. Defaults to None.

        Returns:
            list[paddle.Tensor]: the list tensor shape for cache
        """

        if max_length is None:
            max_length = config.max_sequence_length

        cache_kvs = []
        for _ in range(config.num_hidden_layers):
            cache_kvs.append(
                [
                    2,
                    max_batch_size,
                    config.num_attention_heads // max(config.tensor_parallel_degree, 1),
                    max_length,
                    config.hidden_size // config.num_attention_heads,
                ]
            )
        return cache_kvs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache_kvs,
        seq_len_encoder,
        seq_len_decoder,
        tgt_ids,
        tgt_pos,
        tgt_generation_mask,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        cache = kwargs.get("cache", None)
        pre_caches = kwargs.get("pre_caches", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if cache is not None:
            input_ids = tgt_ids
            position_ids = tgt_pos
            attention_mask = (tgt_generation_mask - 1) * 1e4
            # make inputs_embeds be none in decoder phase.
            # in forward function, it will be assigned according to input_ids.
            inputs_embeds = None
        else:
            attention_mask = (attention_mask - 1) * 1e4
        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "cache_kvs": cache_kvs,
            "seq_len_encoder": seq_len_encoder,
            "seq_len_decoder": seq_len_decoder,
            "cache": cache,
            "pre_caches": pre_caches,
        }
        return model_inputs

    def forward(
        self,
        input_ids:Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        cache=None,
        cache_kvs=None,
        pre_caches=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        #这个是为了保存输入的哦！
        # import numpy as np
        # static_dict = {
        # "input_ids" : input_ids.numpy(),
        # }
        # np.savez('/zhoukangkang/chatglmv2.npz', **static_dict)
        # exit(0)

        transformer_outputs = self.chatglm_v2(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache=cache,
            cache_kvs=cache_kvs,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.chatglm_v2.output_layer(hidden_states)

        # import numpy as np
        # static_dict = {
        # "my" : lm_logits.numpy(),
        # }
        # np.savez('/zhoukangkang/my.npz', **static_dict)
        # exit(0)

        # if (input_ids[0,0] == 906):
        #     import numpy as np
        #     static_dict = {
        #     "my" : lm_logits.numpy(),
        #     }
        #     np.savez('/zhoukangkang/my.npz', **static_dict)
        #     exit(0)


        loss = None
        if labels is not None:
            reshaped_logits = lm_logits.reshape([-1, lm_logits.shape[-1]]).astype("float32")
            reshaped_labels = labels.reshape([-1])

            if self.config.tensor_parallel_degree > 1 and self.config.tensor_parallel_output:
                loss_fn = fleet.meta_parallel.ParallelCrossEntropy()
            else:
                loss_fn = nn.CrossEntropyLoss(reduction="none")

            loss_mask = (labels != -100).astype("float32")
            loss = loss_fn(reshaped_logits, reshaped_labels)
            loss = paddle.sum(loss.reshape([-1]).cast(paddle.float32) * loss_mask.reshape([-1]).cast(paddle.float32))
            loss = loss / loss_mask.sum()

            lm_logits = lm_logits.astype(hidden_states.dtype)
            loss = loss.astype(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

