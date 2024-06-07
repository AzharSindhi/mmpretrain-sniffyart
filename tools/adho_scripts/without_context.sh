#!/usr/bin/env bash

#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --job-name=classifier2_experiments
#SBATCH --gres=gpu:2
#SBATCH --partition=a100,v100
#SBATCH --array=0-14  # Adjust based on the number of experiments
#SBATCH --output=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_err.txt

DIR=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/
cd $DIR

source venv/bin/activate
# default neck: Global average pooling on the final feature map
# without context
commands=(
"tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/hrnet_1"
"tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn50_1"
"tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn101_1"

"tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/hrnet_2"
"tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn50_2"
"tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn101_2"

"tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/hrnet_3"
"tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn50_3"
"tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn101_3"

"tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/hrnet_4"
"tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn50_4"
"tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn101_4"

"tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/hrnet_5"
"tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn50_5"
"tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without_context/default/rn101_5"
)

srun ${commands[$SLURM_ARRAY_TASK_ID]}