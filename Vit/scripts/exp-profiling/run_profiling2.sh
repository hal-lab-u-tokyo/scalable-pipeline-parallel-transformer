#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=2
#PJM -j
#PJM -L jobenv=singularity
#PJM -g gr17
#PJM -L elapse=01:00:00


module load aquarius
module load cuda/12.2
module load ompi-cuda/4.1.5-12.2
module load singularity

# パス設定
PROJECT_ROOT=$PJM_O_WORKDIR
DATASET_DIR="$PROJECT_ROOT/dataset"

echo "Project Root: $PROJECT_ROOT"

# --- 変更点1: 分散学習のためのネットワーク設定 (Wisteria等の場合) ---
# マスターノードの特定（PJM_O_NODEINFの1行目を取得）
MASTER_ADDR=$(head -n 1 $PJM_O_NODEINF)
MASTER_PORT=29500

# --- 変更点2: GPUの可視化設定 (1ノードあたり5枚使う設定) ---
# 各ノードで0~4番のGPUが見えるようにします
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- 実行コマンド ---
# mpiexecを使って、確保した2ノードそれぞれで singularity -> torchrun を起動します
# -np 2: 合計2プロセス（＝2ノード分）
# -npernode 1: 1ノードあたり1つの親プロセス（torchrun）を起動
mpiexec -machinefile $PJM_O_NODEINF -np 2 -npernode 1 \
singularity exec --nv \
--bind $PROJECT_ROOT \
--bind $DATASET_DIR:/dataset \
--env MASTER_ADDR=$MASTER_ADDR \
--env MASTER_PORT=$MASTER_PORT \
"$PROJECT_ROOT/pytorch_env.sif" \
torchrun \
--nnodes=2 \
--nproc_per_node=7 \
--rdzv_id=$PJM_JOBID \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
"$PROJECT_ROOT/main.py" \
--par-mode pp \
--exp-mode=profiling \
--reversible \
--dataset cifar-10 \
--num-hidden-layers 28 \
--hidden-size 768 \
--num-attention-heads 12 \
--batch-size 2800 \
--microbatch-size 50 \
--num-microbatches 56 \
--num-epochs 1 \
--autocast-dtype fp32 \
--lr 1e-7