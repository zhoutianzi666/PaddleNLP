# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/fused_moe.py

"""Fused MoE kernel."""

import functools
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import paddle
import triton
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode
from paddlemix.triton_ops import (
    get_dtype_str,
    paddle_use_triton,
    rendering_common_template,
)

padding_size = 0


@paddle_use_triton(
    key=["1"],
)
def put_along_axis_kernel(A, index, batch, top1: tl.constexpr, top2: tl.constexpr):

    id = tl.program_id(0)
    index_ptr = index + id * top2
    for i in range(top2):
        offset = tl.load(index_ptr + i)
        write_offset = A + id * top1 + offset
        tl.store(write_offset, 1.0)


def put_along_axis_triton_api(A, index):
    op_name = "put_along_axis_triton_api"
    top1 = A.shape[-1]
    top2 = index.shape[-1]
    op_name += f"{top1}_{top2}"
    op_name += f"{get_dtype_str(A.dtype)}"
    op_name += f"{get_dtype_str(index.dtype)}"

    prepare_attr_for_triton_kernel = """
    int batch =  A.shape()[0];
    """

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        prepare_ptr_for_triton_kernel = """
        CUdeviceptr input_ptrs[2] = {
            get_tensor_ptr(A),
            get_tensor_ptr(index)
        };
        """
        template_used = rendering_common_template(
            put_along_axis_triton_api,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
        )
        grid = ("batch",)

        put_along_axis_kernel[(op_name, template_used, grid)](A, index, -1, top1, top2)
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(op_name, A, index)
        return outs[0]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "A": A,
            "index": index,
        }

        useless = helper.create_variable_for_type_inference(dtype="int32")
        outputs = {"useless": useless}
        helper.append_op(
            type=op_name,
            inputs=inputs,
            outputs=outputs,
        )
        return useless


# 可以使用凯伦算子代替
@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Collums of input
    N,
    # Avoid to divide zero
    M,
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


# 可以使用凯伦算子代替
@paddle_use_triton(
    key=["1"],
)
def _per_token_group_quant_fp8_zkk(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    num_rows,
    eps,
    fp8_min,
    fp8_max,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_row = tl.program_id(0) * BLOCK_M
    
    rows = tl.arange(0, BLOCK_M) + start_row
    cols = tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + rows[:, None] * BLOCK_N + cols[None,:]
    mask = rows[:,None] < num_rows

    y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s[:,None], fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)


    y_q_ptrs = y_q_ptr + rows[:, None] * BLOCK_N + cols[None,:]
    tl.store(y_q_ptrs, y_q, mask=mask)
    y_s_ptrs = y_s_ptr + rows
    tl.store(y_s_ptrs, y_s)

def per_token_group_quant_fp8(
    x,
    group_size: int,
    eps: float = 1e-10,
):
    fp8_max, fp8_min = 448.0, -448.0
    x_q = paddle.empty(x.shape, dtype=paddle.float8_e4m3fn)
    M = x.numel() // group_size
    N = group_size
    x_s = paddle.empty(
        [x.shape[0], x.shape[-1] // group_size],
        dtype=paddle.float32,
    )

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        -1,
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


d2s_infer_code = """
std::vector<std::vector<int64_t>> ${op_name}_InferShape(const std::vector<int64_t>& x,
                                                        int64_t group_size,float eps) {
    std::vector<int64_t> x_s_shape = {x[0], x[1] / group_size};
    return {x, x_s_shape};
}
std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
}

"""


def per_token_group_quant_fp8_api(
    x,
    group_size=-1,
    eps=1e-10,
):
    fp8_max, fp8_min = 448.0, -448.0
    assert len(x.shape) == 2

    assert group_size == 128
    BLOCK_N = triton.next_power_of_2(group_size)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    num_stages = 1

    config = {
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    op_name = "per_token_group_quant_fp8_api"
    op_name += f"{get_dtype_str(x.dtype)}"
    op_name += f"_{group_size}"

    prepare_attr_for_triton_kernel = """
    int num_rows = x.shape()[0] * x.shape()[1] / group_size;
    float fp8_max = 448.0;
    float fp8_min = -448.0;
    """

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        x_q = paddle.empty(x.shape, dtype=paddle.float8_e4m3fn)
        x_s = paddle.empty(
            [x.shape[0], x.shape[-1] // group_size],
            dtype=paddle.float32,
        )

        prepare_ptr_for_triton_kernel = """
        auto x_q = paddle::empty(x.shape(), paddle::DataType::FLOAT8_E4M3FN, x.place());
        auto x_s = paddle::empty(
            {x.shape()[0], x.shape()[1] / group_size},
            paddle::DataType::FLOAT32,
            x.place());

        CUdeviceptr input_ptrs[3] = {
            get_tensor_ptr(x),
            get_tensor_ptr(x_q),
            get_tensor_ptr(x_s),
        };
        """
        return_tensor_names = "x_q, x_s"
        template_used = rendering_common_template(
            per_token_group_quant_fp8_api,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
            return_tensor_names,
            d2s_infer_code,
        )
        grid = ("(num_rows + BLOCK_M - 1)/BLOCK_M",)
        BLOCK_M = 8

        _per_token_group_quant_fp8_zkk[(op_name, template_used, grid, [config])](
            x,
            x_q,
            x_s,
            -1,
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N
        )
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(op_name, x, group_size, eps)
        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
        }
        x_q = helper.create_variable_for_type_inference(dtype="float8_e4m3fn")
        x_s = helper.create_variable_for_type_inference(dtype="float32")
        outputs = {"x_q": x_q, "x_s": x_s}
        attrs = {
            "group_size": group_size,
            "eps": eps,
        }
        helper.append_op(
            type=op_name,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )
        return x_q, x_s


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type_enum: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    assert compute_type_enum == 1
    compute_type = tl.bfloat16

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@paddle_use_triton(
    key=["1"],
)
def fused_moe_kernel_zkk(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type_enum: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    assert compute_type_enum == 1
    compute_type = tl.bfloat16

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)

    tl.store(c_ptrs, accumulator, mask=c_mask)


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


@paddle_use_triton(
    key=["1"],
)
def moe_align_block_size_stage1_zkk(
    topk_ids_ptr,
    tokens_cnts_ptr,
    numel,
    tokens_per_thread,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@paddle_use_triton(
    key=["1"],
)
def moe_align_block_size_stage2_zkk(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@paddle_use_triton(
    key=["1"],
)
def moe_align_block_size_stage3_zkk(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@paddle_use_triton(
    key=["1"],
)
def moe_align_block_size_stage4_zkk(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    numel,
    tokens_per_thread,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


# 初步实现，后期可以换成cuda kernel，实现动态or静态
def per_tensor_quant_fp8(x, scale=None):
    x_fp32 = x.cast("float32")
    x_s = x_fp32.abs().max().clip(min=0.000001) / 448.0
    x_q = x_fp32 / x_s
    x_q = x_q.clip(min=-448.0, max=448.0)
    return x_q.cast("float8_e4m3fn"), x_s


def invoke_fused_moe_kernel(
    A,
    B,
    C,  # out
    A_scale,  # a1_scale
    B_scale,  # w1_sacle
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: Optional[List[int]] = None,
) -> None:
    padded_size = 0
    if use_fp8_w8a8:
        assert block_shape is not None
        if block_shape is None:
            A, A_scale = per_tensor_quant_fp8(A, A_scale)
        else:
            block_k = block_shape[1]
            A, A_scale = per_token_group_quant_fp8_api(A, block_k)

            # from paddlenlp_ops import group_quant
            # A, A_scale = group_quant(
            #     A, group_size=128, transpose_scale=False, quant_max_bound=448.0, quant_min_bound=-448.0
            # )
            # assert 128 == block_k

            # A, A_scale = per_token_group_quant_fp8(A, block_k)

            # print((my_scale-A_scale).max().abs())
            # print(my_A.cast("float32") - A.cast("float32"))

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False
    # C1 = paddle.assign(C)

    invoke_fused_moe_kernel_api(
        A,
        B,
        C,  # out
        A_scale,  # a1_scale
        B_scale,  # w1_sacle
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        use_fp8_w8a8,
        use_int8_w8a16,
        even_Ks,
        config,
    )

    assert compute_type == tl.bfloat16
    """
    fused_moe_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel().item(),
        A.strides[0],
        A.strides[1],
        B.strides[0],
        B.strides[2],
        B.strides[1],
        C.strides[1],
        C.strides[2],
        A_scale.strides[0],
        A_scale.strides[1],
        B_scale.strides[0],
        B_scale.strides[2],
        B_scale.strides[1],
        128,
        128,
        MUL_ROUTED_WEIGHT=(int)(mul_routed_weight),
        top_k=top_k,
        compute_type_enum=1,
        use_fp8_w8a8=(int)(use_fp8_w8a8),
        use_int8_w8a16=(int)(use_int8_w8a16),
        even_Ks=(int)(even_Ks),
        **config,
    )
    """

    # assert ((C-C1).abs().max() * 100).cast("int32").item() == 0


def invoke_fused_moe_kernel_api(
    A,
    B,
    C,  # out
    A_scale,  # a1_scale
    B_scale,  # w1_sacle
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight=False,
    top_k=-1,
    use_fp8_w8a8=False,
    use_int8_w8a16=False,
    even_Ks=False,
    config=[],
) -> None:
    prepare_attr_for_triton_kernel = """
            auto N = B.shape()[1];
            auto K = B.shape()[2] - 0;
            auto EM = sorted_token_ids.shape()[0];
            auto num_valid_tokens = (topk_ids.shape()[0]) * (topk_ids.shape()[1]);
            auto stride_am = A.strides()[0];
            auto stride_ak = A.strides()[1];
            auto stride_be = B.strides()[0];
            auto stride_bk = B.strides()[2];
            auto stride_bn = B.strides()[1];
            auto stride_cm = C.strides()[1];
            auto stride_cn = C.strides()[2];

            // auto stride_cm = C.shape()[C.shape().size() - 1];
            // auto stride_cn = 1;

            auto stride_asm = A_scale.strides()[0];
            auto stride_ask = A_scale.strides()[1];
            auto stride_bse = B_scale.strides()[0];
            auto stride_bsk = B_scale.strides()[2];
            auto stride_bsn = B_scale.strides()[1];
    """

    config = {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }
    configs = []

    # for num_warps in [4, 8]:
    #     for block_size_k in [64, 128]:
    #         for block_size_n in [64, 128, 256]:
    #             tmp = dict(config)
    #             tmp["num_warps"] = num_warps
    #             tmp["BLOCK_SIZE_K"] = block_size_k
    #             tmp["BLOCK_SIZE_N"] = block_size_n
    #             configs.append(tmp)
    if B.shape[1] == 256:
        config["BLOCK_SIZE_K"] = 128

    configs.append(dict(config))

    op_name = "fused_moe_zkk"
    op_name += f"{get_dtype_str(A.dtype)}"
    op_name += f"{B.shape[0]}"
    op_name += f"{B.shape[1]}"
    op_name += f"{B.shape[2]}"

    assert config is not None
    for key in config:
        op_name += f"{key}_{config[key]}"

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        prepare_ptr_for_triton_kernel = """
        CUdeviceptr input_ptrs[9] = {
            get_tensor_ptr(A),
            get_tensor_ptr(B),
            get_tensor_ptr(C),
            get_tensor_ptr(A_scale),
            get_tensor_ptr(B_scale),
            get_tensor_ptr(topk_weights),
            get_tensor_ptr(sorted_token_ids),
            get_tensor_ptr(expert_ids),
            get_tensor_ptr(num_tokens_post_padded),
            
        };
        """
        template_used = rendering_common_template(
            invoke_fused_moe_kernel_api,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
        )
        grid = ("(EM+BLOCK_SIZE_M-1)/BLOCK_SIZE_M * ((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N)",)
        padded_size = 0

        assert len(A.shape) == 2
        assert len(C.shape) == 3
        if in_dynamic_or_pir_mode():
            assert C.shape[2] == B.shape[1]

        fused_moe_kernel_zkk[(op_name, template_used, grid, configs)](
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            B.shape[2] - padded_size,
            -1,  # sorted_token_ids.shape[0],
            -1,  # topk_ids.shape[0] * topk_ids.shape[1],
            A.shape[1],  # A.strides[0],
            1,  # A.strides[1],
            B.shape[1] * B.shape[2],  # B.strides[0],
            1,  # B.strides[2],
            B.shape[2],  # B.strides[1],
            B.shape[1],  # C.shape[2], # C.strides[1],
            1,  # C.strides[2],
            A_scale.shape[1],  # A_scale.strides[0] if A_scale is not None and A_scale.dim() == 2 else 0,
            1,  # A_scale.strides[1] if A_scale is not None and A_scale.dim() == 2 else 0,
            B_scale.shape[1]
            * B_scale.shape[2],  # B_scale.strides[0] if B_scale is not None and B_scale.dim() >= 2 else 0,
            1,  # B_scale.strides[2] if B_scale is not None and B_scale.dim() == 3 else 0,
            B_scale.shape[2],  # B_scale.strides[1] if B_scale is not None and B_scale.dim() >= 2 else 0,
            128,
            128,
            MUL_ROUTED_WEIGHT=(int)(mul_routed_weight),
            top_k=top_k,
            compute_type_enum=1,
            use_fp8_w8a8=(int)(use_fp8_w8a8),
            use_int8_w8a16=(int)(use_int8_w8a16),
            even_Ks=(int)(even_Ks),
        )

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name,
            A,
            B,
            C,  # out
            A_scale,  # a1_scale
            B_scale,  # w1_sacle
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            use_fp8_w8a8,
            use_int8_w8a16,
            even_Ks,
        )
        return outs[0]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "A": A,
            "B": B,
            "C": C,
            "A_scale": A_scale,
            "B_scale": B_scale,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "sorted_token_ids": sorted_token_ids,
            "expert_ids": expert_ids,
            "num_tokens_post_padded": num_tokens_post_padded,
        }
        attrs = {
            "mul_routed_weight": mul_routed_weight,
            "top_k": top_k,
            "use_fp8_w8a8": use_fp8_w8a8,
            "use_int8_w8a16": use_int8_w8a16,
            "even_Ks": even_Ks,
        }
        useless = helper.create_variable_for_type_inference(dtype="int32")
        outputs = {"useless": useless}
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
        )
        return useless


def get_device_name(device_id: int = 0) -> str:
    return paddle.device.cuda.get_device_name(device_id)


def get_config_file_name(E: int, N: int, dtype: Optional[str], block_shape: Optional[int] = None) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"


@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = 0,
    block_k: Optional[int] = 0,
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(E, N, dtype, [block_n, block_k])

    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            print("Using configuration from %s for MoE layer.", config_file_path)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    print(
        ("Using default MoE config. Performance might be sub-optimal! " "Config file not found at %s"),
        config_file_path,
    )
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    if dtype == "fp8_w8a8":
        if block_shape is None:
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 32,
                "num_warps": 8,
                "num_stages": 4,
            }
            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 4,
                }
        else:
            # Block-wise quant: BLOCK_SIZE_K must be divisable by block_shape[1]
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 3,
            }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
):
    # from sglang.srt.layers.moe.fused_moe_triton import get_config

    # override_config = get_config()
    # if override_config:
    #     config = override_config
    # else:
    # First try to load optimal config from the file
    E, _, N = w2_shape
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    configs = get_moe_configs(E, N, dtype, block_n, block_k)

    if configs:
        # If an optimal configuration map has been found, look up the
        # optimal config
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Else use the default config
        config = get_default_config(M, E, N, w1_shape[2], top_k, dtype, is_marlin, block_shape)
    # config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3}
    return config


def get_config_dtype_str(
    dtype,
    use_int8_w8a16,
    use_fp8_w8a8,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == paddle.float32:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def fused_experts_impl(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    inplace: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    block_shape: Optional[List[int]] = None,
):
    padded_size = padding_size
    if not use_fp8_w8a8 or block_shape is not None:
        padded_size = 0

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        dtype=hidden_states.dtype,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(128)

    top_k = topk_ids.shape[1]
    assert top_k == 8

    intermediate_cache1 = paddle.empty(
        [M, topk_ids.shape[1], N],
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = paddle.empty(
        (M * topk_ids.shape[1], N // 2),
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = paddle.empty(
        (M, topk_ids.shape[1], w2.shape[1]),
        dtype=hidden_states.dtype,
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == paddle.bfloat16 else tl.float16

    from paddlenlp_ops import preprocess_for_moe

    sorted_token_ids, expert_ids, num_tokens_post_padded = preprocess_for_moe(topk_ids, E, config["BLOCK_SIZE_M"])

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        block_shape=block_shape,
    )

    intermediate_cache2 = paddle.incubate.nn.functional.swiglu(intermediate_cache1.reshape([-1, N]))

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        a2_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        block_shape=block_shape,
    )

    out_hidden_states = paddle.sum(
        intermediate_cache3,
        axis=1,
    )

    return out_hidden_states


def fused_moe(
    hidden_states,
    w1,
    w2,
    scores,
    scores_no_bias,
    topk: int,
    renormalize: bool,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    block_shape: Optional[List[int]] = None,
    refactor: float = 1.0,
    e_score_correction_bias=None,
):
    # Check constraints.
    assert scores.shape[1] == w1.shape[0], "Number of experts mismatch"

    # 经过分组策略后的scores计算topk
    topk_weights, topk_ids = paddle.topk(scores, k=topk, axis=-1, sorted=False)

    if e_score_correction_bias is not None:
        topk_weights = scores_no_bias.take_along_axis(topk_ids, axis=1)

    # renormalize和refactor
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdim=True)

    topk_weights = topk_weights * refactor

    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,
        use_fp8_w8a8,
        use_int8_w8a16,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        block_shape,
    )
