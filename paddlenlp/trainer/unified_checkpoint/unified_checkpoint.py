# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import json
import os

import paddle
from paddle.distributed import fleet

from paddlenlp.peft import LoRAModel, PrefixModelForCausalLM
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.trainer.utils.helper import distributed_file, distributed_isfile
from paddlenlp.transformers.model_utils import (
    PretrainedModel,
    _add_variant,
    load_state_dict,
    unwrap_model,
)
from paddlenlp.transformers.utils import dtype_byte_size
from paddlenlp.utils import infohub
from paddlenlp.utils.env import (
    LORA_WEIGHTS_NAME,
    MAX_QUANTIZATION_TIMES,
    PADDLE_MASTER_WEIGHTS_NAME,
    PADDLE_OPTIMIZER_NAME,
    PADDLE_WEIGHTS_NAME,
    PREFIX_WEIGHTS_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_NAME,
    SAFE_OPTIMIZER_INDEX_NAME,
    SAFE_OPTIMIZER_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_PEFT_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from paddlenlp.utils.log import logger
from paddlenlp.utils.nested import nested_copy

from .async_handler import AsyncCheckpointHandler
from .check_completion import check_unified_checkpoint, check_unified_optimizer
from .load_dynamic import (
    load_unified_checkpoint_dynamically,
    load_unified_optimizer_dynamically,
)
from .load_local import load_unified_checkpoint_locally, load_unified_optimizer_locally
from .load_save_single_card import (
    load_single_card_checkpoint,
    load_single_card_optimizer,
    save_single_card_checkpoint,
    save_single_card_optimizer,
)
from .sharding_split_param_utils import gather_splited_param_for_optimizer
from .utils import (
    FP32_MASTER,
    UnifiedCheckpointOption,
    filter_params,
    gather_sharded_object,
    generate_base_static_name,
    get_expected_state_dict,
    get_sharded_file_name,
    get_sharded_index,
    is_need_master_weight,
    is_sharding_split_param_mode,
    merge_tensor_parallel_for_optimizer,
    merge_tensor_parallel_with_shard,
    reduce_master_weights_status,
    rename_shard_file,
    save_model_config,
)

__all__ = ["UnifiedCheckpointHandler"]


class UnifiedCheckpointHandler:
    def __init__(self, args):
        self.args = args
        self.async_handler = AsyncCheckpointHandler(args)

    def save_unified_checkpoint(self, model, optimizer, output_dir, signal_dir=None):
        """save unified checkpoint

        Args:
            model (PretrainedModel): model to save
            output_dir (str): save dir
            safe_serialization (bool, optional): use safetensors. Defaults to False.

        Raises:
            ValueError: if model is not an instance of `PretrainedModel` and the model cannot be saved
        """
        if isinstance(model, PretrainedModel):
            model_to_save = model
        elif isinstance(unwrap_model(model), PretrainedModel):
            model_to_save = unwrap_model(model)
        elif isinstance(model, PrefixModelForCausalLM) or isinstance(model, LoRAModel):
            model_to_save = model
        else:
            raise ValueError("Unified checkpoint only supports PretrainedModel, LoRAModel and PrefixModelForCausalLM!")

        # Under non distributed environment.
        if paddle.distributed.get_world_size() <= 1:
            save_single_card_checkpoint(model_to_save, output_dir)
            return

        skip_save_model_weight = False
        if UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value in self.args.unified_checkpoint_config:
            if is_need_master_weight(optimizer, is_fp16_or_bp16=(self.args.fp16 or self.args.bf16)):
                logger.info(
                    f"With {UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value}, skip the model checkpoint save."
                    " The master weight will be loaded as model weights for next resumption."
                )
                # not save model weight, load from master weight
                skip_save_model_weight = True

        save_directory = output_dir
        os.makedirs(save_directory, exist_ok=True)
        if signal_dir is not None:
            os.makedirs(signal_dir, exist_ok=True)  # only for async save

        # save model weights
        if not skip_save_model_weight:
            state_dict, shard_file, sharded_index = unified_checkpoint_into_shards(
                self.args, model_to_save, safe_serialization=True
            )
            is_sync_save = True
            if "async_save" in self.args.unified_checkpoint_config:
                is_sync_save = False
            self.async_handler._file_save_async_or_sync(
                state_dict,
                path=os.path.join(save_directory, shard_file),
                signal_path=signal_dir,
                is_sync=is_sync_save,
                state_dict_type="model_weight",
            )
            if sharded_index is not None:
                if isinstance(model_to_save, LoRAModel) or isinstance(model_to_save, PrefixModelForCausalLM):
                    index_name = SAFE_PEFT_WEIGHTS_INDEX_NAME
                else:
                    index_name = SAFE_WEIGHTS_INDEX_NAME
                path = os.path.join(output_dir, index_name)

                if self.args.should_save:
                    with open(path, "w") as f:
                        json.dump(sharded_index, f, indent=4)

        if self.args.should_save:
            save_model_config(model_to_save, save_directory)

        paddle.device.cuda.empty_cache()

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")) and self.args.should_save:
            world_size = paddle.distributed.get_world_size()
            save_info = {
                "world_size": world_size,
                "ignore_save_lr_and_optim": self.args.ignore_save_lr_and_optim,
                "skip_save_model_weight": "skip_save_model_weight" in self.args.unified_checkpoint_config,
                "remove_master_weight": "remove_master_weight" in self.args.unified_checkpoint_config,
            }
            paddle.save(save_info, os.path.join(save_directory, ".saving_info"))

    def load_unified_checkpoint(self, model, resume_from_checkpoint: str):
        """Load potential model checkpoint

        Args:
            model (PretrainedModel): Your model to load
            resume_from_checkpoint (str): path of the checkpoint to load

        Returns:
            None
        """
        if paddle.distributed.get_world_size() <= 1:
            load_single_card_checkpoint(model, resume_from_checkpoint)
            return

        local_resume = check_unified_checkpoint(self.args, model, resume_from_checkpoint, safe_serialization=True)

        if not local_resume:
            logger.info("Begin to dynamically load unified checkpoint!")
            load_unified_checkpoint_dynamically(self.args, model, resume_from_checkpoint, safe_serialization=True)
            return

        if self.args.dataset_rank == 0 or self.args.use_expert_parallel:
            load_unified_checkpoint_locally(self.args, model, resume_from_checkpoint, safe_serialization=True)

    def save_non_merge_optimizer(self, model, optim_state_dict, master_weights, output_dir, signal_dir):
        paddle.device.cuda.empty_cache()

        # gather global master_weights status.
        global_master_weights = reduce_master_weights_status(master_weights is not None)
        if master_weights is None and global_master_weights:
            master_weights = {}

        # get optimizer param mappings
        static2struct_name_mappings = {}
        state_dict = get_expected_state_dict(model)
        for k, v in state_dict.items():
            static2struct_name_mappings[v.name] = k

        # rename optimizer param name
        for key in list(optim_state_dict.keys()):
            static_name, type_name = generate_base_static_name(key)
            new_name = static2struct_name_mappings[static_name] + "/" + type_name
            optim_state_dict[new_name] = optim_state_dict.pop(key)

        if master_weights is not None:
            for key in list(master_weights.keys()):
                master_weights[static2struct_name_mappings[key]] = master_weights.pop(key)

        no_sync_kname = []
        model_state_dict = get_expected_state_dict(model)
        for k, v in model_state_dict.items():
            if getattr(v, "no_sync", False):
                no_sync_kname.append(k)

        hcg = fleet.get_hybrid_communicate_group()
        dp_group = hcg.get_data_parallel_group()
        dp_rank = dp_group.rank if dp_group.nranks > 1 else 0
        if self.args.use_expert_parallel:
            for k in list(optim_state_dict.keys()):
                model_k = k.split("/")[0]
                if dp_rank > 0 and model_k not in no_sync_kname:
                    optim_state_dict.pop(k)
            if master_weights is not None:
                for k in list(master_weights.keys()):
                    model_k = k.split("/")[0]
                    if dp_rank > 0 and model_k not in no_sync_kname:
                        master_weights.pop(k)

        optimizer_name = _add_variant(SAFE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
        master_weights_name = _add_variant(SAFE_MASTER_WEIGHTS_NAME, self.args.optimizer_name_suffix)

        sharded_optim_index = {}
        # save opt index json if checkpoint quantization is on.
        if self.args.ckpt_quant_stage != "O0" and "quant_reach_limit" not in infohub:
            sharded_optim_index["ckpt_quant_stage"] = self.args.ckpt_quant_stage

        sharded_optim_index["quant_ckpt_resume_times"] = (
            infohub["quant_ckpt_resume_times"] if "quant_ckpt_resume_times" in infohub else 0
        )

        if len(sharded_optim_index) > 0:
            optimizer_index_name = SAFE_OPTIMIZER_INDEX_NAME
            path = os.path.join(output_dir, optimizer_index_name)
            if self.args.should_save:
                with open(path, "w") as f:
                    json.dump(sharded_optim_index, f, indent=4)

        is_sync_save = True
        if "async_save" in self.args.unified_checkpoint_config:
            is_sync_save = False
        self.async_handler._file_save_async_or_sync(
            optim_state_dict,
            path=os.path.join(output_dir, optimizer_name),
            signal_path=signal_dir,
            is_sync=is_sync_save,
            state_dict_type="optimizer_weight",
            ckpt_quant_stage=self.args.ckpt_quant_stage if "quant_reach_limit" not in infohub else "O0",
        )
        if master_weights is not None:
            self.async_handler._file_save_async_or_sync(
                master_weights,
                path=os.path.join(output_dir, master_weights_name),
                signal_path=signal_dir,
                is_sync=is_sync_save,
                state_dict_type="master_weight",
            )

    def load_non_merge_optimizer(self, model, optimizer, resume_from_checkpoint, ckpt_quant_stage="O0"):
        # init and get optimizer LR_Scheduler
        returned_optim_state_dict = nested_copy(optimizer.state_dict())

        optimizer_name = _add_variant(SAFE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
        master_weights_name = _add_variant(SAFE_MASTER_WEIGHTS_NAME, self.args.optimizer_name_suffix)
        optimizer_path = os.path.join(resume_from_checkpoint, optimizer_name)
        master_weights_path = os.path.join(resume_from_checkpoint, master_weights_name)
        # no quantization & no master weight represent O1 AMP strategy.
        is_amp_o1 = self.args.fp16_opt_level == "O1"

        model_state_dict = get_expected_state_dict(model)
        struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}  # get optimizer param mappings
        optimizer_state_dict = load_state_dict(
            optimizer_path, None, None, device="expected", ckpt_quant_stage=ckpt_quant_stage
        )
        master_weights = {}
        # normal AMP O2
        if not is_amp_o1 and os.path.isfile(master_weights_path):
            master_weights = load_state_dict(master_weights_path, None, None, device="expected")

        # rename and move to paddle.Tensor
        for key in list(optimizer_state_dict.keys()):
            key_name = key.split("/")
            model_weight_key = key_name[0]
            static_name = struct2static_name_mappings[key_name[0]]
            if not is_amp_o1:
                if model_state_dict[key_name[0]].dtype != paddle.float32:
                    key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
                else:
                    key_name = "_".join([static_name, key_name[1]])
            else:
                key_name = "_".join([static_name, key_name[1]])
            returned_optim_state_dict[key_name] = optimizer_state_dict.pop(key)
            returned_optim_state_dict[key_name].name = key_name

            # master weight cast (only in AMP O2 + remove_master_weight)
            if not is_amp_o1 and not os.path.isfile(master_weights_path):
                master_weights[model_weight_key] = paddle.cast(
                    model_state_dict[model_weight_key], dtype=paddle.float32
                )

        if not is_amp_o1:
            returned_optim_state_dict["master_weights"] = {}
            for key in list(master_weights.keys()):
                static_name = struct2static_name_mappings[key]
                returned_optim_state_dict["master_weights"][static_name] = master_weights.pop(key)
                returned_optim_state_dict["master_weights"][static_name].name = "_".join([static_name, FP32_MASTER])

        return returned_optim_state_dict

    def save_unified_optimizer(self, model, optimizer, output_dir, signal_dir):
        """save unified optimizer

        Args:
            model (PretrainedModel): model used to get key mapping.
            optimizer (Optimizer): optimizer to save
            output_dir (str): Save directory.
            signal_dir (str): Asynchronous saving signal directory.

        """

        if paddle.distributed.get_world_size() <= 1:
            save_single_card_optimizer(model, optimizer, output_dir)  # no need to save signal
            return

        if is_sharding_split_param_mode(self.args):
            optim_state_dict, master_weights = gather_splited_param_for_optimizer(
                optimizer, self.args.ckpt_quant_stage if "quant_reach_limit" not in infohub else "O0"
            )
        else:
            optim_state_dict = nested_copy(optimizer.state_dict())
            master_weights = None
            if "master_weights" in optim_state_dict.keys():
                master_weights = optim_state_dict["master_weights"]
                optim_state_dict.pop("master_weights")
            if "LR_Scheduler" in optim_state_dict.keys():
                optim_state_dict.pop("LR_Scheduler")

        if UnifiedCheckpointOption.REMOVE_MASTER_WEIGHT.value in self.args.unified_checkpoint_config:
            logger.info("Skip master weight saving.")
            master_weights = None

        if "ignore_merge_optimizer" in self.args.unified_checkpoint_config:
            self.save_non_merge_optimizer(model, optim_state_dict, master_weights, output_dir, signal_dir)
            return

        # Split into naive optimizer params and master weights.
        results = unified_optimizer_into_shards(
            self.args, model, optim_state_dict, master_weights, safe_serialization=True
        )
        master_weight_state_dict = None
        if len(results) == 1:
            optim_state_dict, shard_optim_file, sharded_optim_index = results[0]
        else:
            optim_state_dict, shard_optim_file, sharded_optim_index = results[0]
            master_weight_state_dict, shard_master_weight_file, sharded_master_weight_index = results[1]

        paddle.device.cuda.empty_cache()
        save_directory = output_dir
        os.makedirs(save_directory, exist_ok=True)
        if signal_dir is not None:
            os.makedirs(signal_dir, exist_ok=True)

        is_sync_save = True
        if "async_save" in self.args.unified_checkpoint_config:
            is_sync_save = False
        self.async_handler._file_save_async_or_sync(
            optim_state_dict,
            path=os.path.join(save_directory, shard_optim_file),
            signal_path=signal_dir,
            is_sync=is_sync_save,
            state_dict_type="optimizer_weight",
            ckpt_quant_stage=self.args.ckpt_quant_stage if "quant_reach_limit" not in infohub else "O0",
        )
        if master_weight_state_dict is not None:
            self.async_handler._file_save_async_or_sync(
                master_weight_state_dict,
                path=os.path.join(save_directory, shard_master_weight_file),
                signal_path=signal_dir,
                is_sync=is_sync_save,
                state_dict_type="master_weight",
            )

        if sharded_optim_index is not None:
            optimizer_index_name = SAFE_OPTIMIZER_INDEX_NAME
            path = os.path.join(output_dir, optimizer_index_name)
            if self.args.should_save:
                with open(path, "w") as f:
                    json.dump(sharded_optim_index, f, indent=4)

            master_weights_name = SAFE_MASTER_WEIGHTS_INDEX_NAME
            if UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value in self.args.unified_checkpoint_config:
                master_weights_name = SAFE_WEIGHTS_INDEX_NAME
            master_path = os.path.join(output_dir, master_weights_name)
            if master_weight_state_dict is not None:
                if self.args.should_save:
                    with open(master_path, "w") as f:
                        json.dump(sharded_master_weight_index, f, indent=4)

    def load_unified_optimizer(self, model, optimizer, resume_from_checkpoint):
        """Load potential model checkpoint

        Args:
            model (PretrainedModel): Your model to load
            resume_from_checkpoint (str): path of the checkpoint to load

        Returns:
            None
        """

        if paddle.distributed.get_world_size() <= 1:
            optim_state_dict = load_single_card_optimizer(model, optimizer, resume_from_checkpoint)
            return optim_state_dict

        index = {}
        has_merge_optimizer_safetensors = distributed_isfile(
            os.path.join(resume_from_checkpoint, SAFE_OPTIMIZER_INDEX_NAME)
        )
        if has_merge_optimizer_safetensors:
            optimizer_index_file = os.path.join(resume_from_checkpoint, SAFE_OPTIMIZER_INDEX_NAME)
            distributed_file(optimizer_index_file)
            with open(optimizer_index_file, "r") as f:
                index = json.loads(f.read())

        # get quant ckpt info `ckpt_quant_stage` and `quant_ckpt_resume_times`
        ckpt_quant_stage = "O0"
        if "ckpt_quant_stage" in index:
            ckpt_quant_stage = index["ckpt_quant_stage"]

        quant_ckpt_resume_times = 0
        if "quant_ckpt_resume_times" in index:
            quant_ckpt_resume_times = index["quant_ckpt_resume_times"]
        # increment and save resume times in infohub
        if ckpt_quant_stage != "O0":
            quant_ckpt_resume_times += 1
        infohub["quant_ckpt_resume_times"] = quant_ckpt_resume_times

        # Quantization times exceeds the limit. Turn off the quantization strategy.
        if quant_ckpt_resume_times >= MAX_QUANTIZATION_TIMES:
            infohub["quant_reach_limit"] = True
            logger.info("Checkpoint quantization time reach limit and will be closed.")

        # If not having merge optimizer, then load non-merge optimizer.
        if "weight_map" not in index:
            if self.args.data_parallel_rank == 0 or self.args.use_expert_parallel:
                returned_optim_state_dict = self.load_non_merge_optimizer(
                    model,
                    optimizer,
                    resume_from_checkpoint,
                    ckpt_quant_stage=ckpt_quant_stage,
                )
                return returned_optim_state_dict
            else:
                return None

        local_resume = check_unified_optimizer(
            self.args, model, optimizer, resume_from_checkpoint, safe_serialization=True
        )
        if not local_resume:
            logger.info("Begin to dynamically load unified optimizer!")
            returned_optim_state_dict = load_unified_optimizer_dynamically(
                self.args, model, optimizer, resume_from_checkpoint, safe_serialization=True
            )
            return returned_optim_state_dict

        if self.args.data_parallel_rank == 0 or self.args.use_expert_parallel:
            returned_optim_state_dict = load_unified_optimizer_locally(
                self.args, model, optimizer, resume_from_checkpoint, safe_serialization=True
            )
            return returned_optim_state_dict
        return None

    def unlink_shared_memory(self):
        return self.async_handler.unlink_shared_memory()


def unified_checkpoint_into_shards(
    args,
    model_to_save,
    safe_serialization=False,
):
    """Get state_dict and config to save

    Args:
        model_to_save (nn.Layer): model to, save
        safe_serialization (bool, optional): safe serialization using safetensors. Defaults to False.

    Returns:
        tuple: state_dict, config, shard_file: file name, sharded_index: map for weight to file name.
    """
    paddle.device.cuda.empty_cache()
    assert hasattr(model_to_save, "config")

    state_dict = get_expected_state_dict(model_to_save, concat_additional_adapter=True)
    all_filter_keys = filter_params(model_to_save, state_dict, args)

    config_to_save = copy.deepcopy(model_to_save.config)

    if config_to_save.tensor_parallel_degree > 1:
        if isinstance(model_to_save, LoRAModel) or isinstance(model_to_save, PrefixModelForCausalLM):
            tp_actions = model_to_save._get_tensor_parallel_convert_actions(
                all_filter_keys, is_split=False, ignore_error=True
            )
        else:
            tp_actions = model_to_save.get_tensor_parallel_convert_actions(
                model_to_save.config, state_dict.keys(), is_split=False, ignore_error=True
            )
        logger.info("Unified model tensor parallel weights in shards")
        state_dict = merge_tensor_parallel_with_shard(state_dict, tp_actions, all_filter_keys)

    # build index json file
    index_weight_file = {}
    total_size = 0
    if isinstance(model_to_save, LoRAModel):
        weights_name = SAFE_PEFT_WEIGHTS_NAME if safe_serialization else LORA_WEIGHTS_NAME
    elif isinstance(model_to_save, PrefixModelForCausalLM):
        weights_name = SAFE_PEFT_WEIGHTS_NAME if safe_serialization else PREFIX_WEIGHTS_NAME
    else:
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME

    shard_file = get_sharded_file_name(args, weights_name)
    # renumerize shard_file name for expert_parallel.
    if args.use_expert_parallel:
        shard_file = rename_shard_file(args, shard_file, weights_name)

    for key, weight in state_dict.items():
        index_weight_file[key] = shard_file
        total_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    index_file_list, total_size_list = gather_sharded_object(
        index_weight_file, total_size, use_expert_parallel=args.use_expert_parallel
    )
    sharded_index = get_sharded_index(
        index_file_list,
        total_size_list,
    )
    if sharded_index is not None:
        if isinstance(model_to_save, LoRAModel):
            sharded_index["type"] = "lora"
        elif isinstance(model_to_save, PrefixModelForCausalLM):
            sharded_index["type"] = "ptuning"

    paddle.device.cuda.empty_cache()

    return state_dict, shard_file, sharded_index


def unified_optimizer_into_shards(
    args,
    model,
    optim_state_dict,
    master_weights,
    safe_serialization=False,
):
    """Get optimizer state dict and master weight state dict.

    Args:
        optimizer (Optimizer): optimizer to save.
        safe_serialization (bool, optional): safe serialization using safetensors. Defaults to False.
    """
    paddle.device.cuda.empty_cache()

    # gather global master_weights status.
    global_master_weights = reduce_master_weights_status(master_weights is not None)
    if master_weights is None and global_master_weights:
        master_weights = {}

    # get optimizer param mappings
    static2struct_name_mappings = {}
    state_dict = get_expected_state_dict(model)
    fp32_weight = {}

    extra_save_keys = {}
    for k, v in state_dict.items():
        if v.name not in static2struct_name_mappings:
            static2struct_name_mappings[v.name] = k
        else:
            extra_save_keys[v.name] = k
        if master_weights is not None and v.dtype == paddle.float32:
            if args.dataset_rank > 0:  # deal with different dataset rank.
                continue
            fp32_weight[k] = v

    # rename optimizer param
    for key in list(optim_state_dict.keys()):
        static_name, type_name = generate_base_static_name(key)
        new_name = static2struct_name_mappings[static_name] + "/" + type_name
        optim_state_dict[new_name] = optim_state_dict.pop(key)
        if static_name in extra_save_keys:
            extra_new_name = extra_save_keys[static_name] + "/" + type_name
            optim_state_dict[extra_new_name] = optim_state_dict[new_name]

    if master_weights is not None:
        for key in list(master_weights.keys()):
            master_weights[static2struct_name_mappings[key]] = master_weights.pop(key)
            if key in extra_save_keys:
                master_weights[extra_save_keys[key]] = master_weights[static2struct_name_mappings[key]]
        master_weights.update(fp32_weight)

    # filter optimizer param
    if master_weights is not None:
        filter_master_keys = filter_params(model, master_weights, args, is_optimizer=True)
    filter_optim_keys = filter_params(model, optim_state_dict, args, is_optimizer=True)

    tp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
    tp_size = tp_group.nranks

    if tp_size > 1:
        # get tp_actions
        model_keys = []
        for key in optim_state_dict.keys():
            base_model_key = key.split("/")[0]
            if base_model_key not in model_keys:
                model_keys.append(base_model_key)
        if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
            tp_actions = model._get_tensor_parallel_convert_actions(model_keys, is_split=False, ignore_error=True)
        else:
            tp_actions = model.get_tensor_parallel_convert_actions(
                model.config, model_keys, is_split=False, ignore_error=True
            )
        logger.info("Unified optimizer tensor parallel in shards")
        optim_state_dict = merge_tensor_parallel_for_optimizer(
            optim_state_dict,
            tp_actions,
            filter_optim_keys,
            state_dict if args.use_expert_parallel else None,
        )
        paddle.device.cuda.empty_cache()

        if master_weights is not None:
            logger.info("Unified master weight tensor parallel in shards")
            master_weights = merge_tensor_parallel_for_optimizer(
                master_weights,
                tp_actions,
                filter_master_keys,
                state_dict if args.use_expert_parallel else None,
            )
            paddle.device.cuda.empty_cache()

    # build index json file
    index_optimizer_file, index_master_weight_file = {}, {}
    total_optim_size, total_master_weight_size = 0, 0
    optimizer_name = SAFE_OPTIMIZER_NAME if safe_serialization else PADDLE_OPTIMIZER_NAME
    master_weights_name = SAFE_MASTER_WEIGHTS_NAME if safe_serialization else PADDLE_MASTER_WEIGHTS_NAME
    if UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value in args.unified_checkpoint_config:
        master_weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME
    shard_optimizer_file = get_sharded_file_name(args, optimizer_name, is_optimizer=True)
    shard_master_weight_file = get_sharded_file_name(args, master_weights_name, is_optimizer=True)

    for key, weight in optim_state_dict.items():
        index_optimizer_file[key] = shard_optimizer_file
        total_optim_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    if master_weights is not None:
        for key, weight in master_weights.items():
            index_master_weight_file[key] = shard_master_weight_file
            total_master_weight_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    index_optimizer_filelist, total_optim_size_list = gather_sharded_object(
        index_optimizer_file,
        total_optim_size,
        is_optimizer=True,
        use_expert_parallel=args.use_expert_parallel,
    )
    sharded_optim_index = get_sharded_index(index_optimizer_filelist, total_optim_size_list)

    if args.should_save:
        if args.ckpt_quant_stage in ["O1", "O2"] and "quant_reach_limit" not in infohub:
            sharded_optim_index["ckpt_quant_stage"] = args.ckpt_quant_stage
        sharded_optim_index["quant_ckpt_resume_times"] = (
            infohub["quant_ckpt_resume_times"] if "quant_ckpt_resume_times" in infohub else 0
        )

    if master_weights is not None:
        index_master_weight_filelist, total_master_weight_size_list = gather_sharded_object(
            index_master_weight_file,
            total_master_weight_size,
            is_optimizer=True,
            use_expert_parallel=args.use_expert_parallel,
        )
        sharded_master_weight_index = get_sharded_index(index_master_weight_filelist, total_master_weight_size_list)

    if sharded_optim_index is not None:
        if master_weights is not None:
            sharded_optim_index["master_weights"] = True
        else:
            sharded_optim_index["master_weights"] = False

    paddle.device.cuda.empty_cache()
    if master_weights is None:
        return [(optim_state_dict, shard_optimizer_file, sharded_optim_index)]
    else:
        return [
            (optim_state_dict, shard_optimizer_file, sharded_optim_index),
            (master_weights, shard_master_weight_file, sharded_master_weight_index),
        ]
