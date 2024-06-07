#!/usr/bin/env bash

#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --job-name=adho2024_sniffy_with
#SBATCH --gres=gpu:1
#SBATCH --partition=a100,v100
#SBATCH --array=0-29  # Adjust based on the number of experiments
#SBATCH --output=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_err.txt

DIR=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/
cd $DIR

source venv/bin/activate
# default neck: Global average pooling on the final feature map
# without context
commands=(
"python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py work_dirs/adho/with_context/default/hrnet_1/epoch_100.pth --work-dir work_dirs/adho/with_context/default/hrnet_1"
"python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py work_dirs/adho/with_context/default/rn50_1/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn50_1"
"python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py work_dirs/adho/with_context/default/rn101_1/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn101_1"

"python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py work_dirs/adho/with_context/default/hrnet_2/epoch_100.pth --work-dir work_dirs/adho/with_context/default/hrnet_2"
"python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py work_dirs/adho/with_context/default/rn50_2/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn50_2"
"python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py work_dirs/adho/with_context/default/rn101_2/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn101_2"

"python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py work_dirs/adho/with_context/default/hrnet_3/epoch_100.pth --work-dir work_dirs/adho/with_context/default/hrnet_3"
"python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py work_dirs/adho/with_context/default/rn50_3/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn50_3"
"python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py work_dirs/adho/with_context/default/rn101_3/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn101_3"

"python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py work_dirs/adho/with_context/default/hrnet_4/epoch_100.pth --work-dir work_dirs/adho/with_context/default/hrnet_4"
"python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py work_dirs/adho/with_context/default/rn50_4/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn50_4"
"python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py work_dirs/adho/with_context/default/rn101_4/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn101_4"

"python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py work_dirs/adho/with_context/default/hrnet_5/epoch_100.pth --work-dir work_dirs/adho/with_context/default/hrnet_5"
"python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py work_dirs/adho/with_context/default/rn50_5/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn50_5"
"python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py work_dirs/adho/with_context/default/rn101_5/epoch_100.pth --work-dir work_dirs/adho/with_context/default/rn101_5"


)

srun ${commands[$SLURM_ARRAY_TASK_ID]}