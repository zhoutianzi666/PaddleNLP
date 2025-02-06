import paddle
from paddle.distributed import fleet
import paddle.distributed as dist


from paddle.incubate.nn.functional import (
    moe_dispatch,
    moe_ffn,
    moe_reduce,
)

total_cards = 8


strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {"dp_degree": 1, "mp_degree": total_cards, "pp_degree": 1}
fleet.init(is_collective=True, strategy=strategy)

hcg = fleet.get_hybrid_communicate_group()
mp_id = hcg.get_model_parallel_rank()

attention_tp = 4
reduce_group= paddle.distributed.new_group([0, 4])



dtype = "bfloat16"
hidden_size = 2048
moe_intermediate_size = 2816
expert_num = 120
top_k = 4

experts_num_per_gpu = expert_num // total_cards

# 所有的卡的gate_weights是相同的！
gate_weights = paddle.randn([hidden_size, expert_num])
dist.broadcast(gate_weights, src=0)

ffn1_weights = paddle.randn([experts_num_per_gpu, hidden_size, moe_intermediate_size * 2], dtype="bfloat16")
ffn2_weights = paddle.randn([experts_num_per_gpu, moe_intermediate_size, hidden_size], dtype="bfloat16")
ffn1_biases = None
ffn2_biases = None
ffn1_weights_scale =  None
ffn2_weights_scale =  None
quant_type = "None"
norm_topk_prob=False

def compute_ep_moe(tmp_out):
    # 为了少写代码，这里进行了rename
    experts_num = expert_num
    group_num = total_cards // attention_tp
    group_id = mp_id // attention_tp
    first_mp_id_in_group = group_id * attention_tp
    dtype = tmp_out.dtype
    
    IsFirstGPUInAttentionTP = mp_id == first_mp_id_in_group

    gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weights)

    # topk在moe_dispatch中
    (
        permute_input,
        token_nums_per_expert,
        permute_indices_per_token,
        expert_scales_float,
        top_k_indices,
    ) = moe_dispatch(tmp_out, gate_out, top_k, False)

    def get_adjacent_minus(x):
        y = paddle.assign(x)
        y[1:] = y[0:-1]
        y[0] = 0
        y = x - y
        return y

    in_split_sizes_to_permute_input = paddle.zeros([total_cards], token_nums_per_expert.dtype)
    if IsFirstGPUInAttentionTP:
        in_split_sizes_to_permute_input = get_adjacent_minus(token_nums_per_expert) \
                                            .reshape([total_cards, experts_num_per_gpu]) \
                                            .sum(axis=-1)
    else:
        permute_input = paddle.empty([0, hidden_size], dtype=dtype)
    
    
    token_num_in_one_card_from_all_group = None
    if experts_num_per_gpu > 1:
        # 这个代码是为了获得这个token_num_in_one_card_from_all_group!
        if IsFirstGPUInAttentionTP:
            haha = get_adjacent_minus(token_nums_per_expert)
            tmp_in_split_sizes = [experts_num_per_gpu] * total_cards
        else:
            haha = paddle.empty([0], token_nums_per_expert.dtype)
            tmp_in_split_sizes = [0] * total_cards
        tmp_out_split_sizes = ([experts_num_per_gpu] + [0] * (attention_tp - 1)) * group_num

        token_num_in_one_card_from_all_group = paddle.empty([group_num * experts_num_per_gpu], token_nums_per_expert.dtype)
        dist.alltoall_single(token_num_in_one_card_from_all_group, haha, tmp_in_split_sizes, tmp_out_split_sizes)
        token_num_in_one_card_from_all_group = token_num_in_one_card_from_all_group.reshape([group_num, experts_num_per_gpu])
    
  
    
    # 这两个代码就是为了让所有的卡共享token_nums_per_expert

    dist.all_reduce(token_nums_per_expert, group=reduce_group)
    dist.broadcast(token_nums_per_expert, src=0)

    
    start = 0
    start_ep_id = mp_id * experts_num_per_gpu
    if mp_id > 0:
        start = token_nums_per_expert[start_ep_id - 1]
    end = token_nums_per_expert[start_ep_id + experts_num_per_gpu - 1]
    token_nums_per_expert_per_card = token_nums_per_expert[start_ep_id:start_ep_id + experts_num_per_gpu] - start

    permute_input_per_card = paddle.empty([end-start, hidden_size], dtype=dtype)

    out_split_sizes_to_permute_input = paddle.zeros_like(in_split_sizes_to_permute_input)
    dist.alltoall(out_split_sizes_to_permute_input, in_split_sizes_to_permute_input)
    dist.alltoall_single(permute_input_per_card, permute_input, in_split_sizes_to_permute_input, out_split_sizes_to_permute_input)
    
    def run_permute_input(permute_input_per_card, flag=True):
        if experts_num_per_gpu == 1:
            return permute_input_per_card
        new_permute_input_per_card = paddle.empty_like(permute_input_per_card)
        j = 0
        for i in range(experts_num_per_gpu):
            for group_id in range(group_num):
                num = token_num_in_one_card_from_all_group[group_id][i]
                j1 = token_num_in_one_card_from_all_group[0:group_id].sum()
                j1 += token_num_in_one_card_from_all_group[group_id][0:i].sum()
                if num > 0:
                    if flag:
                        new_permute_input_per_card[j:j+num] = permute_input_per_card[j1:j1+num]
                    else:
                        new_permute_input_per_card[j1:j1+num] = permute_input_per_card[j:j+num]
                j += num
        return new_permute_input_per_card
    
    permute_input_per_card = run_permute_input(permute_input_per_card)

    ffn_out = moe_ffn(
        permute_input_per_card,
        token_nums_per_expert_per_card,
        ffn1_weights,
        ffn2_weights,
        ffn1_biases,
        ffn1_weights_scale,
        ffn2_weights_scale,
        quant_type,
    )

    ffn_out = run_permute_input(ffn_out, False)
    dist.alltoall_single(permute_input, ffn_out, out_split_sizes_to_permute_input, in_split_sizes_to_permute_input)
    noise_pred = paddle.assign(permute_input)

    fused_moe_out = moe_reduce(
        noise_pred,
        expert_scales_float,
        permute_indices_per_token,
        top_k_indices,
        ffn2_biases,
        norm_topk_prob,
    )

    return fused_moe_out


if __name__ == '__main__':
    tmp_out = paddle.randn([82, hidden_size], dtype)

    for i in range(10):
        res = compute_ep_moe(tmp_out)
    print(res)


