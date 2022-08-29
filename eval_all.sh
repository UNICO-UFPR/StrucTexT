#!/bin/bash

export FLAGS_allocator_strategy=auto_growth
export FLAGS_init_allocated_mem=True
export FLAGS_eager_delete_scope=True

export FLAGS_eager_delete_tensor_gb=2
export FLAGS_memory_fraction_of_eager_deletion=1.0

export FLAGS_fraction_of_cpu_memory_to_use=0.1
export FLAGS_fraction_of_gpu_memory_to_use=0.2

eval_checkpoint () {
    m=$1
    ds=$2

    .venv/bin/python tools/eval_infer.py \
        --config_file configs/base/pick_config.json \
        --task_type labeling_segment \
        --weights_path $m.pdparams \
        --label_path datasets/real-full/label \
        --image_path datasets/real-full/images \
        --out_path predictions/$m && \
    .venv/bin/python analysis.py predictions/$m
}



# real - real
eval_checkpoint real datasets/real-part && \
# synth - real
for m in dup25 dup50 full inst25 inst50; do
    eval_checkpoint $m datasets/real-full
done
