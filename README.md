# Scalable Pipeline-Parallel Training of Transformer with Batch Expansion Using Reversible Computation

[![Conference](https://img.shields.io/badge/IEEE%20ICPADS%202026-Full%20Paper-blue.svg)](https://icpads2026.github.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0%20%2F%20MIT-green.svg)]()

This repository contains the official implementation of the paper:
> **"Scalable Pipeline-Parallel Training of Transformer with Batch Expansion Using Reversible Computation"**  
> *Shunta Seki, et al.*  
> Accepted as a **Full Paper** at **The 32nd IEEE International Conference on Parallel and Distributed Systems (ICPADS 2026)**.

---

## 📖 Overview

Training large Transformer models (such as BERT and Vision Transformer) poses significant challenges due to surging GPU memory requirements. While **Pipeline Parallelism (PP)** partitions model layers across multiple GPUs to enable distributed training, standard PP suffers from three fundamental bottlenecks:
1. **Pipeline Bubbles (Idle Time)**: The startup (fill) and drain phases in pipeline execution introduce idle periods on GPUs, degrading hardware utilization.
2. **Activation Memory Bottleneck**: Retaining intermediate activations for the backward pass restricts the maximum number of micro-batches that can fit into GPU memory without causing Out-of-Memory (OOM) errors.
3. **Overhead of Recomputation**: Conventional Activation Checkpointing (CP) saves memory but incurs considerable forward recomputation overhead (~33%), which constrains overall training throughput.

### 💡 Proposed Method: Reversible Pipeline Parallelism with Batch Expansion

To address these challenges, this framework introduces:

1. **Reversible Computation for Activation Memory Elimination**:
   - Transformer layers are reformulated using reversible residual blocks ($Y_1 = X_1 + \mathcal{F}(X_2)$, $Y_2 = X_2 + \mathcal{G}(Y_1)$).
   - Intermediate activations are reconstructed on the fly from output activations ($X_2 = Y_2 - \mathcal{G}(Y_1)$, $X_1 = Y_1 - \mathcal{F}(X_2)$) during the backward pass instead of caching them in GPU memory during the forward pass.
   - This drastically slashes activation memory from $O(L)$ to $O(1)$ without the redundant forward recomputation required by checkpointing.

2. **Batch Expansion (Micro-Batch Scaling)**:
   - By leveraging the memory budget freed by reversible computation, we scale up the number of micro-batches ($M$) within the same GPU memory capacity.
   - Expanding micro-batches drastically shrinks the pipeline bubble ratio ($\frac{P-1}{M + P - 1}$, where $P$ is the number of pipeline stages), achieving substantial **throughput acceleration** and higher GPU utilization.

3. **Pareprop (Parallel Backward Propagation)**:
   - Decouples weight gradient computation (`backward_weight`) from activation reconstruction and input gradient propagation (`backward_input`), overlapping communication and computation across pipeline stages.

---

## 📂 Repository Structure

The codebase is organized into two primary subdirectories: **BERT experiments (`BERT/`)** for Natural Language Processing and **Vision Transformer experiments (`Vit/`)** for Computer Vision. Each directory is fully self-contained with its own model definitions, pipeline execution engine, and experiment pipelines.

```text
.
├── README.md                  # This documentation
│
├── BERT/                      # BERT (Natural Language Processing) experiments
│   ├── main.py                # Main CLI entry point for BERT
│   ├── args.py                # Argument parsing for models, parallel modes, and schedules
│   ├── config/                # Configuration directory
│   ├── environment/           # Environment setup & container definition
│   │   └── pytorch.def        # Singularity/Apptainer definition file (PyTorch 2.6.0 + CUDA 12.4)
│   ├── experiment/            # Experiment workflows
│   │   ├── base.py            # Base class for experiment pipelines (BaseExperiment)
│   │   ├── profiling/         # GPU memory, step latency, and pipeline bubble profiling
│   │   │   ├── main.py        # Profiling execution
│   │   │   ├── plot.py        # Visualizing memory and time breakdown
│   │   │   └── memory_predictor.py # Theoretical memory estimation model
│   │   ├── actv_err/          # Numerical accuracy verification for reversible reconstruction
│   │   │   ├── main.py        # Reconstruction error measurement
│   │   │   └── plot.py        # Error plotting
│   │   └── training/          # End-to-end model training and convergence tracking
│   ├── models/                # Model architecture definitions
│   │   ├── reversible.py      # Core reversible residual building blocks (ReversibleBlock / ReversibleSequence)
│   │   ├── bert_blocks.py     # BERT sub-modules (Self-Attention, Intermediate, Embeddings)
│   │   ├── bert_stages.py     # Pipeline stages for Reversible BERT
│   │   ├── normal_bert_stages.py # Standard (baseline) BERT pipeline stages
│   │   └── bert_stages_with_inverse.py # Custom inverse backward stages
│   ├── pipelining/            # Custom pipeline parallelism runtime (extends torch.distributed.pipelining)
│   │   ├── stage.py           # Pipeline stage base abstraction
│   │   ├── reversible_stage.py # Stage implementation for reversible computation
│   │   ├── reversible_stage_with_pareprop.py # Stage implementation with Pareprop backward overlap
│   │   ├── stage_with_cp.py   # Activation Checkpointing (CP) baseline stage
│   │   ├── schedules.py       # Pipeline schedule implementations (e.g., 1F1B)
│   │   ├── microbatch.py      # Micro-batch chunking and collation utilities
│   │   └── _backward.py       # Low-level backward gradient engine
│   ├── scripts/               # Automation scripts
│   │   ├── prepare_dataset.py # IMDb dataset download & tokenizer preparation
│   │   ├── exp-profiling/     # Batch cluster job scripts (e.g., Wisteria/Aquarius PJM)
│   │   ├── exp-actv_err/      # Shell scripts for activation error verification
│   │   └── exp-training/      # Shell scripts for model training
│   ├── src/                   # Core shared framework
│   │   ├── config.py          # GlobalConfig data classes & hyperparameter definitions
│   │   ├── engine.py          # Training and evaluation engine
│   │   ├── env.py             # Distributed process group initialization (PP, DDP, FSDP)
│   │   ├── dataloader.py      # Distributed DataLoader setup
│   │   ├── dataset_factory.py # Dataset loaders
│   │   ├── dataset_registry.py# Registry of available datasets
│   │   ├── ckpt_io.py         # Checkpointing save/load logic
│   │   ├── logger.py / logging_utils.py # Logging utilities
│   │   └── metric_tracker.py  # Metrics logging (loss, throughput, memory peak)
│   └── tests/                 # Unit tests & smoke tests
│
└── Vit/                       # Vision Transformer (Computer Vision) experiments
    ├── main.py                # Main CLI entry point for ViT
    ├── args.py                # Argument parsing (supports CIFAR-10/100, CINIC-10, ImageNet-1K)
    ├── config/                # Configuration directory
    ├── environment/           # Environment setup & container definition (pytorch.def)
    ├── experiment/            # Experiment workflows (profiling, actv_err, training)
    ├── models/                # ViT model architectures
    │   ├── reversible.py      # Reversible residual building blocks
    │   ├── vit_blocks.py      # PatchEmbedding, Multi-Head Attention, MLP
    │   ├── vit_stages.py      # Pipeline stages for Reversible ViT
    │   └── normal_vit_stages.py # Standard ViT pipeline stages
    ├── pipelining/            # Custom pipeline runtime (reversible, Pareprop, 1F1B schedules)
    ├── scripts/               # Experiment scripts (profiling and training on vision datasets)
    ├── src/                   # Core shared framework (engine, loaders, distributed env)
    └── tests/                 # Smoke tests
```

---

## 🛠️ Subdirectory Breakdown & Responsibilities

### 1. `BERT/` (BERT Experiments)
- **Target Architecture**: Transformer Encoder (equivalent to BERT-Base / BERT-Large).
- **Target Dataset**: IMDb sentiment classification and synthetic debug datasets.
- **Key Modules**:
  - `BERT/models/reversible.py`: Defines `ReversibleBlock` and `ReversibleSequence` enabling memory-efficient forward and analytical backward reconstruction.
  - `BERT/models/bert_stages.py`: Partitions encoder layers across stages for pipeline parallel schedules.
  - `BERT/models/normal_bert_stages.py`: Standard non-reversible BERT pipeline baseline.

### 2. `Vit/` (Vision Transformer Experiments)
- **Target Architecture**: Vision Transformer (ViT-Base / ViT-Large).
- **Target Datasets**: CIFAR-10, CIFAR-100, CINIC-10, and ImageNet-1K.
- **Key Modules**:
  - `Vit/models/vit_blocks.py`: Vision Transformer building blocks (Patch Embedding, Multi-Head Self-Attention, MLP).
  - `Vit/models/vit_stages.py`: Stage partition implementation for 2D image token sequences.
  - `Vit/models/normal_vit_stages.py`: Standard ViT pipeline baseline.

### 3. `pipelining/` (Custom Pipeline Runtime)
Built upon PyTorch's distributed pipelining framework (`torch.distributed.pipelining`):
- `reversible_stage.py`: Discards intermediate forward activations and performs reverse reconstruction during the backward pass.
- `reversible_stage_with_pareprop.py`: Implements **Pareprop** (Parallel Backward Propagation) by separating backward execution into input gradient propagation and weight gradient accumulation to overlap computation.
- `stage_with_cp.py`: Activation Checkpointing (CP) baseline implementation.
- `schedules.py`: Pipeline schedule orchestration (1F1B, etc.).

### 4. `experiment/` (Experiment Suites)
- **`profiling/`**: Measures GPU peak memory allocation, step execution time, communication overhead, and idle bubble time. Supports plotting (`plot.py`) and comparison against theoretical memory predictions (`memory_predictor.py`).
- **`actv_err/`**: Quantifies numerical reconstruction drift between forward activations and reversibly reconstructed activations across layers in FP32, FP16, and BF16.
- **`training/`**: Executes actual model training, reporting training loss and validation accuracy curves.

---

## ⚙️ Requirements & Setup

### Prerequisites
- Linux (Ubuntu 20.04+ recommended) or macOS
- Python 3.10+
- CUDA 12.4+ (for multi-GPU cluster runs)
- PyTorch 2.6.0+

### Python Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.53.3 datasets==4.4.1 accelerate==1.9.0 pandas matplotlib scikit-learn
```

### Apptainer / Singularity Container
For supercomputer / HPC environments (e.g., Wisteria/BDEC-01 Aquarius at the University of Tokyo):
```bash
singularity build pytorch_env.sif BERT/environment/pytorch.def
```

---

## 🚀 Usage & Quick Start

### 1. Dataset Preparation
Prepare the IMDb dataset for BERT experiments:
```bash
python BERT/scripts/prepare_dataset.py --outdir ./dataset
```

### 2. Running BERT Experiments

#### (A) Profiling (Memory & Speedup with Batch Expansion)
Run pipeline-parallel profiling with reversible computation and Pareprop:
```bash
torchrun --nproc_per_node=8 BERT/main.py \
    --exp-mode profiling \
    --par-mode pp \
    --reversible \
    --is_pareprop \
    --dataset imdb \
    --num-hidden-layers 24 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 512 \
    --batch-size 512 \
    --microbatch-size 16 \
    --num-microbatches 32 \
    --autocast-dtype fp32
```

#### (B) Activation Reconstruction Error Verification
```bash
torchrun --nproc_per_node=8 BERT/main.py \
    --exp-mode actv-err \
    --par-mode pp \
    --reversible \
    --dataset imdb \
    --batch-size 128
```

#### (C) End-to-End Model Training
```bash
torchrun --nproc_per_node=8 BERT/main.py \
    --exp-mode training \
    --par-mode pp \
    --reversible \
    --dataset imdb \
    --batch-size 256 \
    --num-microbatches 16 \
    --num-epochs 3 \
    --lr 1e-4
```

### 3. Running Vision Transformer (ViT) Experiments

```bash
torchrun --nproc_per_node=8 Vit/main.py \
    --exp-mode profiling \
    --par-mode pp \
    --reversible \
    --is_pareprop \
    --dataset cifar-10 \
    --num-hidden-layers 12 \
    --hidden-size 768 \
    --batch-size 512 \
    --microbatch-size 16 \
    --num-microbatches 32 \
    --num-epochs 15 \
    --autocast-dtype fp32 \
    --lr 1e-5
```

---

## 📋 Command-Line Arguments Reference

| Argument | Type / Choices | Default | Description |
| :--- | :--- | :--- | :--- |
| `--exp-mode`, `-e` | `training`, `profiling`, `actv-err`, `acc` | `training` | Experiment pipeline mode to execute |
| `--par-mode` | `none`, `pp`, `ddp`, `fsdp` | `none` | Parallelization strategy (`pp` for pipeline parallelism) |
| `--reversible` | flag | `False` | Enable reversible residual blocks |
| `--is_pareprop` | flag | `False` | Enable Pareprop (Parallel Backward Propagation) optimization |
| `--checkpointing`| flag | `False` | Enable standard activation checkpointing (for baseline comparison) |
| `--batch-size` | int | `256` | Global batch size per training step |
| `--microbatch-size` | int | `64` | Batch size per micro-batch |
| `--num-microbatches`| int | `4` | Number of micro-batches in pipeline schedule |
| `--autocast-dtype` | `fp32`, `fp16`, `bf16` | `fp32` | Autocast floating-point precision |
| `--dataset` | `imdb`, `cifar-10`, `cifar-100`, etc. | Model-dependent | Training / evaluation dataset |

---

## 📜 Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{seki2026scalable,
  title={Scalable Pipeline-Parallel Training of Transformer with Batch Expansion Using Reversible Computation},
  author={Seki, Shunta and others},
  booktitle={Proceedings of the 32nd IEEE International Conference on Parallel and Distributed Systems (ICPADS)},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 License

This project is licensed under Apache-2.0 / MIT. See individual file headers and `LICENSE` for details.