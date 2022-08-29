#!/bin/bash

export FLAGS_allocator_strategy=auto_growth
export FLAGS_init_allocated_mem=True
export FLAGS_eager_delete_scope=True

export FLAGS_eager_delete_tensor_gb=4
export FLAGS_memory_fraction_of_eager_deletion=1.0

export FLAGS_fraction_of_cpu_memory_to_use=0.1
export FLAGS_fraction_of_gpu_memory_to_use=0.7

.venv/bin/python trainer.py \
    -c configs/base/pick_config.json \
    -d datasets/real-part \
    -o full.pdparams -bs 8 -g 0
