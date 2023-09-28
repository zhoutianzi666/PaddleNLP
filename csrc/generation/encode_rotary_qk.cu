// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"

template <typename T>
__global__ void NeoXRotaryKernel(const T *input,
                                 const float *cos_emb,
                                 const float *sin_emb,
                                 const int *sequence_lengths,
                                 T *output,
                                 const int rotary_emb_dims,
                                 const int batch_size,
                                 const int head_num,
                                 const int seq_len,
                                 const int last_dim) {
  int bi = blockIdx.x;
  int hi = blockIdx.y;
  int si = blockIdx.z;
  if (sequence_lengths && si >= sequence_lengths[bi] * rotary_emb_dims) return;
  int half_lastdim = last_dim / 2;
  for (int ti = threadIdx.x; ti < half_lastdim; ti += blockDim.x) {
    int base_idx = bi * head_num * seq_len * last_dim +
                   hi * seq_len * last_dim + si * last_dim;
    int left_idx = base_idx + ti;
    const int right_idx = base_idx + ti + half_lastdim;
    int emb_idx_left = bi * seq_len * last_dim + si * last_dim + ti;
    int emb_idx_right =
        bi * seq_len * last_dim + si * last_dim + ti + half_lastdim;
    float input_left = static_cast<float>(input[left_idx]);
    float input_right = static_cast<float>(input[right_idx]);

    float cos_tmp_left = cos_emb[emb_idx_left];
    float sin_tmp_left = sin_emb[emb_idx_left];
    float cos_tmp_right = cos_emb[emb_idx_right];
    float sin_tmp_right = sin_emb[emb_idx_right];

    T res1 =
        static_cast<T>(input_left * cos_tmp_left - input_right * sin_tmp_left);
    T res2 = static_cast<T>(input_right * cos_tmp_right +
                            input_left * sin_tmp_right);
    output[left_idx] = res1;
    output[right_idx] = res2;
  }
}


template <typename T>
__global__ void RotaryKernel(const T *input,
                             const float *cos_emb,
                             const float *sin_emb,
                             const int *sequence_lengths,
                             T *output,
                             const int rotary_emb_dims,
                             const int batch_size,
                             const int head_num,
                             const int seq_len,
                             const int last_dim) {
  int bi = blockIdx.x;
  int hi = blockIdx.y;
  int si = blockIdx.z;
  if (sequence_lengths && si >= sequence_lengths[bi] * rotary_emb_dims) return;
  int half_lastdim = last_dim / 2;
  int rotary_last_dim = last_dim / 4;

  // Note(ZhenyuLi): Calculate the relevant data at one time, so that no
  // additional space is required.
  for (int ti = threadIdx.x; ti < half_lastdim; ti += blockDim.x) {
    
    if (ti >= half_lastdim / 2) break;
    
    int base_idx = bi * head_num * seq_len * last_dim +
                   hi * seq_len * last_dim + si * last_dim;
    int left_idx = base_idx + 2 * ti;
    const int right_idx = base_idx + 2 * ti + 1;
    
    int emb_idx = bi * seq_len * rotary_last_dim + si * rotary_last_dim + ti;
    
    float input_left = static_cast<float>(input[left_idx]);
    float input_right = static_cast<float>(input[right_idx]);
    float cos_tmp = cos_emb[emb_idx];
    float sin_tmp = sin_emb[emb_idx];
    T res1 = static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
    T res2 = static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    output[left_idx] = res1;
    output[right_idx] = res2;
  }
}

template <paddle::DataType D>
void LaunchRotaryQK(const paddle::Tensor& q, 
                    const paddle::Tensor& kv, 
                    const paddle::Tensor& rotary_emb, 
                    const paddle::Tensor& seq_lens, 
                    const int32_t rotary_emb_dims, 
                    bool use_neox) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;


    const int32_t batch_size = q.shape()[0];
    const int32_t head_num = q.shape()[1];
    const int32_t seq_len = q.shape()[2];
    const int32_t dim_head = q.shape()[3];

    auto cu_stream = q.stream();
    // 卧槽，每个block计算 last_dim 这么多个数字！
    dim3 grid(batch_size, head_num, seq_len * rotary_emb_dims);
    const int last_dim = dim_head / rotary_emb_dims;
    auto getBlockSize = [](int dim) {
        if (dim > 256) {
        return 512;
        } else if (dim > 128) {
        return 256;
        } else if (dim > 64) {
        return 128;
        } else if (dim > 32) {
        return 64;
        } else {
        return 32;
        }
    };
    // last_dim / 2 因为要算两个数字啊！
    int BlockSize = getBlockSize(last_dim / 2);
    const float *cos_emb = rotary_emb.data<float>();
    // batch_size are always 1!
    const float *sin_emb = rotary_emb.data<float>() + 1 * seq_len * dim_head / 4;

    // std::cout << "batch_size * seq_len * dim_head" << batch_size * seq_len * dim_head << std::endl;
    
    const DataType_* q_data = reinterpret_cast<const DataType_*>(q.data<data_t>()); 
    const DataType_* k_data = reinterpret_cast<const DataType_*>(kv.data<data_t>()); 

    DataType_* q_out_data = reinterpret_cast<DataType_*>(const_cast<data_t*>(q.data<data_t>())); 
    DataType_* k_out_data = reinterpret_cast<DataType_*>(const_cast<data_t*>(kv.data<data_t>())); 


    if (!use_neox) {
        RotaryKernel<<<grid, BlockSize, 0, cu_stream>>>(
            q_data,
            cos_emb,
            sin_emb,
            seq_lens.data<int>()/*sequence_lengths*/,
            q_out_data,
            rotary_emb_dims,
            batch_size,
            head_num,
            seq_len * rotary_emb_dims,
            last_dim);
        RotaryKernel<<<grid, BlockSize, 0, cu_stream>>>(
            k_data,
            cos_emb,
            sin_emb,
            seq_lens.data<int>()/*sequence_lengths*/,
            k_out_data,
            rotary_emb_dims,
            batch_size,
            head_num,
            seq_len * rotary_emb_dims,
            last_dim);
    } else {
        NeoXRotaryKernel<<<grid, BlockSize, 0, cu_stream>>>(
            q_data,
            cos_emb,
            sin_emb,
            seq_lens.data<int>()/*sequence_lengths*/,
            q_out_data,
            rotary_emb_dims,
            batch_size,
            head_num,
            seq_len * rotary_emb_dims,
            last_dim);
        NeoXRotaryKernel<<<grid, BlockSize, 0, cu_stream>>>(
            k_data,
            cos_emb,
            sin_emb,
            seq_lens.data<int>()/*sequence_lengths*/,
            k_out_data,
            rotary_emb_dims,
            batch_size,
            head_num,
            seq_len * rotary_emb_dims,
            last_dim);
    }
}

void RotaryQK(const paddle::Tensor& q, 
              const paddle::Tensor& kv, 
              const paddle::Tensor& rotary_emb, 
              const paddle::Tensor& seq_lens,
              const int32_t rotary_emb_dims, 
              bool use_neox) {
    switch (q.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchRotaryQK<paddle::DataType::BFLOAT16>(
                q, kv, rotary_emb, seq_lens, rotary_emb_dims, use_neox
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchRotaryQK<paddle::DataType::FLOAT16>(
                q, kv, rotary_emb, seq_lens, rotary_emb_dims, use_neox
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchRotaryQK<paddle::DataType::FLOAT32>(
                q, kv, rotary_emb, seq_lens, rotary_emb_dims, use_neox
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
        }
    }
}



PD_BUILD_OP(encode_rotary_qk)
    .Inputs({"q", "kv", "rotary_emb", "seq_lens"})
    .Outputs({"rotary_q_out", "rotary_kv_out"})
    .SetInplaceMap({{"q", "rotary_q_out"}, {"kv", "rotary_kv_out"}})
    .Attrs({"rotary_emb_dims: int", "use_neox: bool"})
    .SetKernelFn(PD_KERNEL(RotaryQK)); 