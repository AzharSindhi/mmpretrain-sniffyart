#!/usr/bin/env bash

#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=train_sumac
#SBATCH --gres=gpu:1
#SBATCH --partition=a100,v100
#SBATCH --array=0-14  # Adjust based on the number of experiments
#SBATCH --output=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_err.txt

commands=(
"tools/dist_train.sh projects/dsp-sumac/hrnet-w32_4xb32_dsp_pt.py 2 --work-dir work_dirs/pt_hrnet_1"
"tools/dist_train.sh projects/dsp-sumac/resnet50_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn50_1"
"tools/dist_train.sh projects/dsp-sumac/resnet101_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn101_1"


"tools/dist_train.sh projects/dsp-sumac/hrnet-w32_4xb32_dsp_pt.py 2 --work-dir work_dirs/pt_hrnet_2"
"tools/dist_train.sh projects/dsp-sumac/resnet50_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn50_2"
"tools/dist_train.sh projects/dsp-sumac/resnet101_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn101_2"

"tools/dist_train.sh projects/dsp-sumac/hrnet-w32_4xb32_dsp_pt.py 2 --work-dir work_dirs/pt_hrnet_3"
"tools/dist_train.sh projects/dsp-sumac/resnet50_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn50_3"
"tools/dist_train.sh projects/dsp-sumac/resnet101_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn101_3"

"tools/dist_train.sh projects/dsp-sumac/hrnet-w32_4xb32_dsp_pt.py 2 --work-dir work_dirs/pt_hrnet_4"
"tools/dist_train.sh projects/dsp-sumac/resnet50_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn50_4"
"tools/dist_train.sh projects/dsp-sumac/resnet101_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn101_4"

"tools/dist_train.sh projects/dsp-sumac/hrnet-w32_4xb32_dsp_pt.py 2 --work-dir work_dirs/pt_hrnet_5"
"tools/dist_train.sh projects/dsp-sumac/resnet50_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn50_5"
"tools/dist_train.sh projects/dsp-sumac/resnet101_8xb32_dsp_pt.py 2 --work-dir work_dirs/pt_rn101_5"

)

srun ${commands[$SLURM_ARRAY_TASK_ID]}