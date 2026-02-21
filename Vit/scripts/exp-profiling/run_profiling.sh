#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=01:00:00
#PJM -g gr17
#PJM -j
#PJM -L jobenv=singularity

module load aquarius cuda singularity

# パス設定
PROJECT_ROOT=$PJM_O_WORKDIR
DATASET_DIR="$PROJECT_ROOT/dataset"

echo "Project Root: $PROJECT_ROOT"

# 実行コマンド
singularity exec --nv \
--bind $PROJECT_ROOT \
--bind $DATASET_DIR:/dataset \
"$PROJECT_ROOT/pytorch_env.sif" \
torchrun --nproc_per_node=8 "$PROJECT_ROOT/main.py" \
--par-mode pp \
--reversible \
--exp-mode=profiling \
--dataset cifar-10 \
--num-hidden-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--batch-size 512 \
--microbatch-size 16 \
--num-microbatches 32 \
--num-epochs 1 \
--autocast-dtype fp32 \
--lr 1e-7
EOF