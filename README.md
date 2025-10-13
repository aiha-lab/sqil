
# SQIL: Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control

[![Conference](https://img.shields.io/badge/ICCV-2025-blue.svg?style=for-the-badge)]()
[![Paper](https://img.shields.io/badge/arXiv-2505.15304-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2505.15304)
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-009688?style=for-the-badge)](https://aiha-lab.github.io/sqil/)


---

## Introduction

This repository contains the **implementation of
*Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control (SQIL)***, accepted to **ICCV 2025**.

<p align="center">
  <img src="https://aiha-lab.github.io/sqil/assets/fig_1.PNG" alt="SQIL Teaser" width="85%">
</p>
  <p class="teaser-summary"style="width: 100%; margin: 0.0em auto;">
    <strong>SQIL</strong> is the first systematic study of <strong>Quantized Imitation Learning</strong>, revealing that most quantized failures occur at <strong><em>mission-critical states</em></strong> requiring fine-grained control. 
    By leveraging <strong>policy-driven saliency (SIS)</strong> and a <strong>SIS-weighted 4-bit QAT</strong> scheme, SQIL achieves <strong>2&ndash;4&times;</strong> efficiency gains while preserving <strong>full-precision-level success rates</strong> across real-world robotics, autonomous driving and physics simulation.
  </p>

---

## Installation


```bash
# Create and activate environment
conda create -n sqil python=3.10 -y
conda activate sqil

# Install PyTorch (adjust CUDA version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Clone repository
git clone https://github.com/aiha-lab/sqil.git
cd sqil
pip install -e .

# (Optional) for Flash-Attention 2
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```
## LIBERO Setup

Clone and install the LIBERO repo:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```
Additionally, install other required packages (the same as OpenVLA):

```bash
cd sqil
pip install -r experiments/robot/libero/libero_requirements.txt
```
To download the modified versions of the LIBERO datasets that we used in our fine-tuning experiments, run the command below.
This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (Long) datasets in RLDS data format (~10 GB total).


```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```
---

## Usage

## (Prerequisite) Prepare a Low-Precision Policy (AWQ / QuaRot)

Before running **Precompute Saliency Importance Scores**, you need a **low-precision (quantized) policy**.  
Please use an existing repo such as:

- **AWQ**: https://github.com/mit-han-lab/llm-awq  
- **QuaRot** https://github.com/spcl/QuaRot

### 1️⃣ Precompute Saliency Importance Scores

Compute per-sample **State Importance Scores** using a frozen teacher VLA.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_sqil.py \
  --precompute True \
  --data_root_dir <PATH/TO/RLDS> \
  --dataset_name <DATASET_NAME> \
  --saliency_cache_path <PATH/TO/SAVE/sis_cache.jsonl> \
  --batch_size 16 \
  --image_aug False
```

---

### 2️⃣ SQIL : QAT + Quantization-Robust action Distillation (QRD) 

Fine-tune a quantized VLA using cached SIS scores to modulate imitation loss.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_sqil.py \
  --precompute False \
  --vla_path <PATH/TO/QUANTIZED_VLA> \
  --data_root_dir <PATH/TO/RLDS> \
  --dataset_name <DATASET_NAME> \
  --saliency_cache_path <PATH/TO/SAVE/sis_cache.jsonl> \
  --run_root_dir <PATH/TO/RUNS> \
  --adapter_tmp_dir <PATH/TO/ADAPTER_TMP> \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --image_aug True \
  --wandb_project sqil \
  --wandb_entity <YOUR_WANDB_ID>
```

## Evaluation on LIBERO

After fine-tuning, you can evaluate the resulting **SQIL policy** on the LIBERO benchmarks
using the same evaluation pipeline as OpenVLA.

We provide evaluation scripts under `experiments/robot/libero/`.

```bash
# Example: evaluate SQIL-tuned policy on LIBERO

# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <PATH/TO/YOUR_SQIL_CHECKPOINT_SPATIAL> \
  --task_suite_name libero_spatial \
  --center_crop True

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <PATH/TO/YOUR_SQIL_CHECKPOINT_OBJECT> \
  --task_suite_name libero_object \
  --center_crop True

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <PATH/TO/YOUR_SQIL_CHECKPOINT_GOAL>\
  --task_suite_name libero_goal \
  --center_crop True

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <PATH/TO/YOUR_SQIL_CHECKPOINT_LONG>\
  --task_suite_name libero_10 \
  --center_crop True
```

---

## File Structure

```
sqil/
├── vla-scripts/
│   └── finetune_sqil.py         # main fine-tuning entry
├── prismatic/                   # OpenVLA / Prismatic modules
└── datasets/                    # RLDS-formatted datasets
```

---




## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{park2025sqil,
  title     = {Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control},
  author    = {Park, Seongmin and Kim, Hyungmin and Kim, Sangwoo and Jeon, Wonseok and Yang, Juyoung and Jeon, Byeongwook and Oh, Yoonseon and Choi, Jungwook},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

---

## Acknowledgements

SQIL builds upon [OpenVLA](https://github.com/openvla/openvla)
and [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms).
We thank the Open X-Embodiment and LIBERO teams for datasets and simulation benchmarks.

---
