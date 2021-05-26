#!/usr/bin/env python3

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

r"""
To configure model on Triton, you can use `config_model_on_triton.py` script.
This will prepare layout of Model Repository, including  Model Configuration.

```shell script
python ./triton/config_model_on_triton.py \
    --model-repository /model_repository \
    --model-path /models/exported/model.onnx \
    --model-format onnx \
    --model-name ResNet50 \
    --model-version 1 \
    --max-batch-size 32 \
    --precision fp16 \
    --backend-accelerator trt \
    --load-model explicit \
    --timeout 120 \
    --verbose
```

If Triton server to which we prepare model repository is running with **explicit model control mode**,
use `--load-model` argument to send request load_model request to Triton Inference Server.
If server is listening on non-default address or port use `--server-url` argument to point server control endpoint.
If it is required to use HTTP protocol to communicate with Triton server use `--http` argument.

To improve inference throughput you can use
[dynamic batching](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#dynamic-batcher)
for your model by providing `--preferred-batch-sizes` and `--max-queue-delay-us` parameters.

For models which doesn't support batching, set `--max-batch-sizes` to 0.

By default Triton will [automatically obtain inputs and outputs definitions](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#auto-generated-model-configuration).
but for TorchScript ang TF GraphDef models script uses file with I/O specs. This file is automatically generated
when the model is converted to ScriptModule (either traced or scripted).
If there is a need to pass different than default path to I/O spec file use `--io-spec` CLI argument.

I/O spec file is yaml file with below structure:

```yaml
- inputs:
  - name: input
    dtype: float32   # np.dtype name
    shape: [None, 224, 224, 3]
- outputs:
  - name: probabilities
    dtype: float32
    shape: [None, 1001]
  - name: classes
    dtype: int32
    shape: [None, 1]
```

"""

import argparse
import logging
import time

from model_navigator import Accelerator, Format, Precision
from model_navigator.args import str2bool
from model_navigator.log import set_logger, log_dict
from model_navigator.triton import ModelConfig, TritonClient, TritonModelStore

LOGGER = logging.getLogger("config_model")


def _available_enum_values(my_enum):
    return [item.value for item in my_enum]


def main():
    parser = argparse.ArgumentParser(
        description="Create Triton model repository and model configuration", allow_abbrev=False
    )
    parser.add_argument("--model-repository", required=True, help="Path to Triton model repository.")
    parser.add_argument("--model-path", required=True, help="Path to model to configure")

    # TODO: automation
    parser.add_argument(
        "--model-format",
        required=True,
        choices=_available_enum_values(Format),
        help="Format of model to deploy",
    )
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--model-version", default="1", help="Version of model (default 1)")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size allowed for inference. "
        "A max_batch_size value of 0 indicates that batching is not allowed for the model",
    )
    # TODO: automation
    parser.add_argument(
        "--precision",
        type=str,
        default=Precision.FP16.value,
        choices=_available_enum_values(Precision),
        help="Model precision (parameter used only by Tensorflow backend with TensorRT optimization)",
    )

    # Triton Inference Server endpoint
    parser.add_argument(
        "--server-url",
        type=str,
        default="grpc://localhost:8001",
        help="Inference server URL in format protocol://host[:port] (default grpc://localhost:8001)",
    )
    parser.add_argument(
        "--load-model",
        choices=["none", "poll", "explicit"],
        help="Loading model while Triton Server is in given model control mode",
    )
    parser.add_argument(
        "--timeout", default=120, help="Timeout in seconds to wait till model load (default=120)", type=int
    )

    # optimization related
    parser.add_argument(
        "--backend-accelerator",
        type=str,
        choices=_available_enum_values(Accelerator),
        default=Accelerator.TRT.value,
        help="Select Backend Accelerator used to serve model",
    )
    parser.add_argument("--number-of-model-instances", type=int, default=1, help="Number of model instances per GPU")
    parser.add_argument(
        "--preferred-batch-sizes",
        type=int,
        nargs="*",
        help="Batch sizes that the dynamic batcher should attempt to create. "
        "In case --max-queue-delay-us is set and this parameter is not, default value will be --max-batch-size",
    )
    parser.add_argument(
        "--max-queue-delay-us",
        type=int,
        default=0,
        help="Max delay time which dynamic batcher shall wait to form a batch (default 0)",
    )
    parser.add_argument(
        "--capture-cuda-graph",
        type=int,
        default=0,
        help="Use cuda capture graph (used only by TensorRT platform)",
    )

    parser.add_argument("-v", "--verbose", help="Provide verbose logs", type=str2bool, default=False)
    args = parser.parse_args()

    set_logger(verbose=args.verbose)
    log_dict("args", vars(args))

    config = ModelConfig.create(
        model_path=args.model_path,
        # model definition
        model_name=args.model_name,
        model_version=args.model_version,
        model_format=args.model_format,
        precision=args.precision,
        max_batch_size=args.max_batch_size,
        # optimization
        accelerator=args.backend_accelerator,
        gpu_engine_count=args.number_of_model_instances,
        preferred_batch_sizes=args.preferred_batch_sizes or [],
        max_queue_delay_us=args.max_queue_delay_us,
        capture_cuda_graph=args.capture_cuda_graph,
    )

    model_store = TritonModelStore(args.model_repository)
    model_store.deploy_model(model_config=config, model_path=args.model_path)

    if args.load_model != "none":
        client = TritonClient(server_url=args.server_url, verbose=args.verbose)
        client.wait_for_server_ready(timeout=args.timeout)

        if args.load_model == "explicit":
            client.load_model(model_name=args.model_name)

        if args.load_model == "poll":
            time.sleep(15)

        client.wait_for_model(model_name=args.model_name, model_version=args.model_version, timeout_s=args.timeout)


if __name__ == "__main__":
    main()
