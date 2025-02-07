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

import os
import unittest
from tempfile import TemporaryDirectory

from parameterized import parameterized

from paddlenlp.mergekit import MergeConfig, MergeModel
from paddlenlp.transformers import AutoModel
from tests.testing_utils import require_gpu


class TestMergeModel(unittest.TestCase):
    @parameterized.expand([("slerp",), ("della",), ("dare_linear",), ("ties",)])
    def test_merge_model(self, merge_method):
        with TemporaryDirectory() as tempdir:
            model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert", dtype="bfloat16")
            pd_path = os.path.join(tempdir, "pd_model")
            model.save_pretrained(pd_path)
            safe_path = os.path.join(tempdir, "safe_model")
            model.save_pretrained(safe_path, safe_serialization="safetensors")

            # test mix
            merge_config = MergeConfig(
                merge_method=merge_method, model_path_list=[safe_path, pd_path], output_path=tempdir
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test mix with base model
            merge_config = MergeConfig(
                merge_method=merge_method,
                model_path_list=[safe_path, pd_path],
                output_path=tempdir,
                base_model_path=safe_path,
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test safetensor only
            merge_config = MergeConfig(
                merge_method=merge_method, model_path_list=[safe_path, safe_path], output_path=tempdir, n_process=2
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test safetensor only with base model
            merge_config = MergeConfig(
                merge_method=merge_method,
                model_path_list=[safe_path, safe_path],
                output_path=tempdir,
                n_process=2,
                base_model_path=safe_path,
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

            # test safetensor only with pd
            merge_config = MergeConfig(
                merge_method=merge_method,
                model_path_list=[safe_path, safe_path],
                output_path=tempdir,
                n_process=2,
                tensor_type="pd",
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()

    @parameterized.expand([("slerp",), ("della",), ("dare_linear",), ("ties",)])
    @require_gpu(2)
    def test_merge_model_gpu(self, merge_method):
        with TemporaryDirectory() as tempdir:
            model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert", dtype="bfloat16")
            pd_path = os.path.join(tempdir, "pd_model")
            model.save_pretrained(pd_path)
            safe_path = os.path.join(tempdir, "safe_model")
            model.save_pretrained(safe_path, safe_serialization="safetensors")

            # test safetensor only with pd and gpu
            merge_config = MergeConfig(
                merge_method=merge_method,
                model_path_list=[safe_path, safe_path],
                output_path=tempdir,
                device="gpu",
                tensor_type="pd",
            )
            mergekit = MergeModel(merge_config)
            mergekit.merge_model()
