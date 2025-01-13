# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import time

import numpy as np
import paddle
import paddlenlp_ops

np.random.seed(2024)


def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
    cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens_this_time)
    cum_offsets = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_offsets[1:] = cum_offsets_now
    token_num = paddle.sum(seq_lens_this_time)
    padding_offsets = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i]
        cum_offset = cum_offsets[i]
        for j in range(seq_len_now):
            padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
        cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1]
        cu_seqlens_q[i + 1] = cum_seq_len
        cu_seqlens_k[i + 1] = cum_seq_len
    return padding_offsets, cum_offsets[:-1], cu_seqlens_q, cu_seqlens_k


run_time = 1
warm_up = 0

block_size = 64
head_dim = 256

max_dec_len = 1
num_q_head = 128  # 14
num_kv_head = 128  # 14
dtype = "bfloat16"
use_neox_rotary_style = False

# prefill
max_length = 8192
bsz = 1
input_length = 2000
seq_lens_enc = [
    input_length,
] * bsz
seq_lens_dec = [
    0,
] * bsz
seq_lens_this_time = seq_lens_enc
max_enc_len_this_time = max(seq_lens_enc)
max_dec_len_this_time = max(seq_lens_dec)
max_enc_len_this_time = paddle.to_tensor([max_enc_len_this_time], "int32", place=paddle.CPUPlace())
max_dec_len_this_time = paddle.to_tensor([max_dec_len_this_time], "int32", place=paddle.CPUPlace())
token_num = sum(seq_lens_this_time)

block_num_per_seq = (max_length + block_size - 1) // block_size
max_block_num = block_num_per_seq * bsz
free_list = list(range(max_block_num - 1, -1, -1))

block_tables = paddle.zeros(shape=(bsz, block_num_per_seq), dtype="int32")

for i in range(bsz):
    need_block_num = (seq_lens_enc[i] + max_dec_len + block_size - 1) // block_size
    for j in range(need_block_num):
        block_id = free_list.pop()
        block_tables[i, j] = block_id


seq_lens_encoder = paddle.to_tensor(seq_lens_enc, "int32")
seq_lens_this_time = paddle.to_tensor(seq_lens_this_time, "int32")
seq_lens_decoder = paddle.to_tensor(seq_lens_dec, "int32")

padding_offsets, cum_offsets, cu_seqlens_q, cu_seqlens_k = get_padding_offset(bsz, max_length, seq_lens_this_time)

q_varlen_shape = [token_num, (num_q_head + 2 * num_kv_head) * head_dim]
cache_shape = (
    max_block_num,
    num_kv_head,
    block_size,
    head_dim,
)
rotary_embs_shape = [2, max_length, 1, head_dim if use_neox_rotary_style else head_dim // 2]
qkv_bias_shape = [num_q_head + 2 * num_kv_head, head_dim]

encoder_block_shape_q = 64
decoder_block_shape_q = 16
max_partition_size = 512
encoder_max_partition_size = 32768
(
    encoder_batch_ids,
    encoder_tile_ids_per_batch,
    encoder_num_blocks,
    kv_batch_ids,
    kv_tile_ids_per_batch,
    kv_num_blocks,
    decoder_batch_ids,
    decoder_tile_ids_per_batch,
    decoder_num_blocks,
    max_len_kv,
) = paddlenlp_ops.get_block_shape_and_split_kv_block(
    seq_lens_encoder,
    seq_lens_decoder,
    max_enc_len_this_time,
    max_dec_len_this_time,
    seq_lens_this_time,
    cum_offsets,
    num_q_head // num_kv_head,
    block_size,
    1,
)

# def gqa_attention(qkv,
#                     cache_k,
#                     cache_v,
#                     seq_lens_encoder,
#                     seq_lens_decoder,
#                     seq_lens_this_time,
#                     padding_offsets,
#                     cum_offsets,
#                     kv_num_head):
#     q, k, v = paddle.split(qkv, [num_q_head, num_kv_head, num_kv_head], axis=1)
#     query_group = paddle.chunk(q, chunks=kv_num_head, axis=2)

#     scale_qk_coeff = self.head_dim**0.5
#     attn_mask = paddle.zeros(shape=[seq_len, seq_len], dtype="float32")
#     for i in range(seq_len):
#         for j in range(seq_len):
#             if i <= j:
#                 attn_mask[i][j] = -10000

#     for query in query_group:
#         query = query.scale(1.0 / scale_qk_coeff)
#         score = paddle.matmul(x=query, y=k, transpose_y=True)
#         score += attn_mask[:seq_lens_encoder][:seq_lens_encoder]
#         score = paddle.softmax(score, axis=-1)
#         out = paddle.matmul(score, v)

#     product = paddle.matmul(x=q.scale(1.0 / scale_qk_coeff), y=k, transpose_y=True)

#     weights = product +

#     out = paddle.matmul(weights, v)
# def base_attention():
#     qkv = paddle.randn(shape=q_varlen_shape).astype(dtype)
#     cache_k = paddle.ones(shape=cache_shape).astype(dtype)
#     cache_v = paddle.ones(shape=cache_shape).astype(dtype)
#     rotary_embs = paddle.randn(shape=rotary_embs_shape).astype("float32")
#     qkv_scale = None
#     qkv_bias = paddle.randn(shape=qkv_bias_shape).astype(dtype)

#     cache_k_scale = paddle.ones(shape=[num_kv_head]).astype(dtype)
#     cache_v_scale = paddle.ones(shape=[num_kv_head]).astype(dtype)
#     cache_k_out_scale = paddle.ones(shape=[num_kv_head]).astype(dtype)
#     cache_v_out_scale = paddle.ones(shape=[num_head_dim]).astype(dtype)
#     shift_bias = paddle.zeros(shape=[num_q_head, head_dim]).astype(dtype)
#     smooth_weight = paddle.ones(shape=[num_q_head, head_dim]).astype(dtype)
#     no_tensor = paddle.zeros(shape=[1]).astype("int32")
#     s_time = 0
#     for i in range(run_time + warm_up):
#         if i == warm_up:

#             s_time = time.time()
#         out = naive_attention(
#                 qkv,
#                 cache_k,
#                 cache_v,
#                 seq_lens_encoder,
#                 seq_lens_decoder,
#                 seq_lens_this_time,
#                 padding_offsets,
#                 cum_offsets,
#                 block_tables,
#                 encoder_batch_ids,
#                 encoder_tile_ids_per_batch)


def test_append_c16_attention():
    qkv = paddle.randn(shape=q_varlen_shape).astype(dtype)
    cache_k = paddle.ones(shape=cache_shape).astype(dtype)
    cache_v = paddle.ones(shape=cache_shape).astype(dtype)
    rotary_embs = paddle.randn(shape=rotary_embs_shape).astype("float32")
    qkv_bias = paddle.randn(shape=qkv_bias_shape).astype(dtype)

    shift_bias = paddle.zeros(shape=[num_q_head, head_dim]).astype(dtype)
    smooth_weight = paddle.ones(shape=[num_q_head, head_dim]).astype(dtype)
    s_time = 0
    for i in range(run_time + warm_up):
        if i == warm_up:
            s_time = time.time()
        out = paddlenlp_ops.append_attention(
            qkv,
            cache_k,
            cache_v,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            padding_offsets,
            cum_offsets,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            max_enc_len_this_time,
            max_dec_len_this_time,
            max_len_kv,
            rotary_embs,
            None,  # attn_mask
            qkv_bias,
            None,  # qkv_scale
            None,
            None,
            None,
            None,
            None,  # cache_k_zp
            None,  # cache_v_zp
            shift_bias,
            smooth_weight,
            "bf16",
            "none",  # cache_quant_type
            use_neox_rotary_style,
            max_length,
            127.0,
            -127.0,
            0.0,  # out_linear_in_scale
            1,  # speculate_max_draft_token_num
            True,  # causal
            False,  # speculate_decoder
        )
        paddle.device.synchronize
    e_time = time.time()
    print("output shape is : ", out.shape)
    print("prefill c16 attention cost_time: {} ms".format((e_time - s_time) / run_time * 1000))


if __name__ == "__main__":
    test_append_c16_attention()
