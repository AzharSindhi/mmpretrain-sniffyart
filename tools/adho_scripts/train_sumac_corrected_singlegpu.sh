#!/usr/bin/env bash

#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=train_sumac
#SBATCH --gres=gpu:1
#SBATCH --partition=a100,v100
#SBATCH --array=0-14  # Adjust based on the number of experiments
#SBATCH --output=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_err.txt

CLASS_WEIGHTS="[0.0073,0.1866,.0543,.4594,.0591,.0406,.1926]"

commands=(
  "python tools/train.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/hrnet_1 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet50_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn50_1 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet101_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn101_1 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"

  "python tools/train.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/hrnet_2 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet50_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn50_2 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet101_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn101_2 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"

  "python tools/train.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/hrnet_3 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet50_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn50_3 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet101_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn101_3 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"

  "python tools/train.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/hrnet_4 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet50_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn50_4 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet101_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn101_4 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"

  "python tools/train.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/hrnet_5 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet50_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn50_5 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
  "python tools/train.py projects/dsp-sumac/resnet101_8xb32_dsp.py --work-dir work_dirs/sumac_corrected_singlegpu/without_context/default/rn101_5 --cfg-options model.head.loss.class_weight=$CLASS_WEIGHTS"
)

srun ${commands[$SLURM_ARRAY_TASK_ID]}
