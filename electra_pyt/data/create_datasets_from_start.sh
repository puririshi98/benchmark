#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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
#DATA_PREP_WORKING_DIR=/workspace/electra/pretraining_data/data_prep

# Download
#python3 /workspace/bert/data/dataPrep.py --action download --dataset bookscorpus
#python3 /workspace/bert/data/dataPrep.py --action download --dataset wikicorpus_en

#python3 /workspace/bert/data/dataPrep.py --action download --dataset google_pretrained_weights  # Includes vocab

#All other pretraining related option commented out since only fine-tuning is supported at the moment.
#python3 /workspace/bert/data/dataPrep.py --action download --dataset squad
#python3 /workspace/bert/data/dataPrep.py --action download --dataset mrpc


# Properly format the text files
#python3 /workspace/bert/data/dataPrep.py --action text_formatting --dataset bookscorpus
#python3 /workspace/bert/data/dataPrep.py --action text_formatting --dataset wikicorpus_en


# Shard the text files (group wiki+books then shard)
python3 /workspace/electra/data/dataPrep.py --action sharding --dataset books_wiki_en_corpus \
  --n_test_shards 2048 --n_training_shards 2048

# Create hdf5 files Phase 1
python3 /workspace/electra/data/dataPrep.py --action create_hdf5_files --dataset books_wiki_en_corpus --max_seq_length 128


# Create hdf5 files Phase 2
python3 /workspace/electra/data/dataPrep.py --action create_hdf5_files --dataset books_wiki_en_corpus --max_seq_length 512

