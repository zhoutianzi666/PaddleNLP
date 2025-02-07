import paddle
from paddle.distributed import fleet
import paddle.distributed as dist


from paddle.incubate.nn.functional import (
    moe_dispatch,
    moe_ffn,
    moe_reduce,
    swiglu,
)

total_cards = 8


strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {"dp_degree": 1, "mp_degree": total_cards, "pp_degree": 1}
fleet.init(is_collective=True, strategy=strategy)

hcg = fleet.get_hybrid_communicate_group()
mp_id = hcg.get_model_parallel_rank()

attention_tp = 2

dtype = "bfloat16"
hidden_size = 8192
moe_intermediate_size = 3584
experts_num = 8
top_k = 8

experts_num_per_gpu = experts_num // total_cards

# 所有的卡的gate_weights是相同的！
gate_weights = paddle.randn([hidden_size, experts_num])
dist.broadcast(gate_weights, src=0)

ffn1_weights = paddle.randn([experts_num_per_gpu, hidden_size, moe_intermediate_size * 2], dtype="bfloat16")
ffn2_weights = paddle.randn([experts_num_per_gpu, moe_intermediate_size, hidden_size], dtype="bfloat16")
ffn1_biases = None
ffn2_biases = None
ffn1_weights_scale =  None
ffn2_weights_scale =  None
quant_type = "None"
norm_topk_prob=False


one_ffn1_weights = ffn1_weights[0]
one_ffn2_weights = ffn2_weights[0]


group_num = total_cards // attention_tp
group_id = mp_id // attention_tp
first_mp_id_in_group = group_id * attention_tp
IsFirstGPUInAttentionTP = mp_id == first_mp_id_in_group

def compute_baseline(tmp_out):
    all_tensor = []
    dist.all_gather(all_tensor, ffn1_weights, group=fleet.get_hybrid_communicate_group().get_model_parallel_group())
    all_ffn1_weights = paddle.concat(all_tensor, axis=0)
    all_tensor = []
    dist.all_gather(all_tensor, ffn2_weights, group=fleet.get_hybrid_communicate_group().get_model_parallel_group())
    all_ffn2_weights = paddle.concat(all_tensor, axis=0)

    gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weights)

    (
        permute_input,
        token_nums_per_expert,
        permute_indices_per_token,
        expert_scales_float,
        top_k_indices,
    ) = moe_dispatch(tmp_out, gate_out, top_k, False)

    
    ffn_out = None
    if False:
        ffn_out = moe_ffn(
            permute_input,
            token_nums_per_expert,
            ffn1_weights,
            ffn2_weights,
            ffn1_biases,
            ffn1_weights_scale,
            ffn2_weights_scale,
            quant_type,
        )
    else:
        for i in range(experts_num):
            weight_A = all_ffn1_weights[i]
            weight_B = all_ffn2_weights[i]
            start = 0
            if i > 0:
                start = token_nums_per_expert[i-1]
            end = token_nums_per_expert[i]
            x = permute_input[start:end]

            tmp_out1 = paddle.matmul(x, weight_A)
            tmp_out1 = swiglu(tmp_out1)
            ffn_out = paddle.matmul(tmp_out1, weight_B)
            permute_input[start:end] = ffn_out
        ffn_out = paddle.assign(permute_input)

    fused_moe_out = moe_reduce(
        ffn_out,
        expert_scales_float,
        permute_indices_per_token,
        top_k_indices,
        ffn2_biases,
        norm_topk_prob,
    )
    return fused_moe_out



start_event = paddle.device.Event(enable_timing=True)
end_event = paddle.device.Event(enable_timing=True)

def compute_ep_moe(tmp_out):
    assert experts_num_per_gpu == 1
    dtype = tmp_out.dtype
    
    start_event.record()

    gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weights)

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"gate_compute: {round(elapsed_time_ms,2)} ms")

    start_event.record()

    (
        permute_input,
        token_nums_per_expert,
        permute_indices_per_token,
        expert_scales_float,
        top_k_indices,
    ) = moe_dispatch(tmp_out, gate_out, top_k, False)


    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"moe_dispatch: {round(elapsed_time_ms,2)} ms")

    def get_adjacent_minus(x):
        y = paddle.assign(x)
        y[1:] = y[0:-1]
        y[0] = 0
        y = x - y
        return y
    
    start_event.record()
    act_in_split_size = paddle.zeros([total_cards], token_nums_per_expert.dtype)
    
    if IsFirstGPUInAttentionTP:
        act_in_split_size = get_adjacent_minus(token_nums_per_expert)
    else:
        permute_input = paddle.empty([0, hidden_size], dtype=dtype)
    
    act_out_split_size = paddle.empty_like(act_in_split_size)
    
    dist.alltoall(act_out_split_size, act_in_split_size)
    this_card_token_nums = act_out_split_size.sum().reshape([1])

    permute_input_per_card = paddle.empty([this_card_token_nums, hidden_size], dtype=dtype)
    dist.alltoall_single(permute_input_per_card, permute_input, act_in_split_size, act_out_split_size)

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"2alltoall: {round(elapsed_time_ms,2)} ms")

    start_event.record()

    tmp_out1 = paddle.matmul(permute_input_per_card, one_ffn1_weights)
    tmp_out1 = swiglu(tmp_out1)
    ffn_out = paddle.matmul(tmp_out1, one_ffn2_weights)

    # ffn_out = moe_ffn(permute_input_per_card, this_card_token_nums, ffn1_weights, ffn2_weights, ffn1_biases, ffn1_weights_scale, ffn2_weights_scale, quant_type)
    
    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"ffn: {round(elapsed_time_ms,2)} ms")

    start_event.record()
    dist.alltoall_single(permute_input, ffn_out, act_out_split_size, act_in_split_size,sync_op=True)
    moe_reduce_input = paddle.assign(permute_input)

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"1alltoall: {round(elapsed_time_ms,2)} ms")

    start_event.record()

    fused_moe_out = moe_reduce(
        moe_reduce_input,
        expert_scales_float,
        permute_indices_per_token,
        top_k_indices,
        ffn2_biases,
        norm_topk_prob,
    )

    end_event.record()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"moe_reduce: {round(elapsed_time_ms,2)} ms")

    return fused_moe_out

if __name__ == '__main__':


    for i in range(20):
        flush_cache = paddle.randn([512, 512, 512], dtype)

        tmp_x = paddle.randn([128, hidden_size], dtype)
        tmp_x = tmp_x * 0.01
        res = compute_ep_moe(tmp_x)
    
    baseline = compute_baseline(tmp_x)

    if res.shape[0] > 0:
        print(res.shape)
        print(baseline.shape)
        print(paddle.max(paddle.abs(res-baseline)))













# data = paddle.ones([256, hidden_size])
# all_tensor = []
# dist.all_gather(all_tensor, data)
# dist.all_gather(all_tensor, data)

# start.record()

# dist.all_gather(all_tensor, data)

# end.record()
# elapsed_time_ms = start_event.elapsed_time(end_event)
# print(f"dist.all_gather: {round(elapsed_time_ms,2)} ms")

# print(data)




