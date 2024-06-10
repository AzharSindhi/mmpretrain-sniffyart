# #!/usr/bin/env bash

# #!/bin/bash
# #SBATCH --time=00:10:00
# #SBATCH --job-name=test_sumac
# #SBATCH --gres=gpu:1
# #SBATCH --partition=a100,v100
# #SBATCH --array=0-14  # Adjust based on the number of experiments
# #SBATCH --output=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_out.txt
# #SBATCH --error=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_err.txt

# DIR=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/
# cd $DIR

# source venv/bin/activate

# commands=(
#   "python tools/test.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/hrnet_1/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/hrnet_1"
#   "python tools/test.py projects/dsp-sumac/resnet50_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn50_1/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn50_1"
#   "python tools/test.py projects/dsp-sumac/resnet101_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn101_1/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn101_1"

#   "python tools/test.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/hrnet_2/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/hrnet_2"
#   "python tools/test.py projects/dsp-sumac/resnet50_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn50_2/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn50_2"
#   "python tools/test.py projects/dsp-sumac/resnet101_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn101_2/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn101_2"

#   "python tools/test.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/hrnet_3/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/hrnet_3"
  "python tools/test.py projects/dsp-sumac/resnet50_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn50_3/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn50_3"
#   "python tools/test.py projects/dsp-sumac/resnet101_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn101_3/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn101_3"

#   "python tools/test.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/hrnet_4/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/hrnet_4"
#   "python tools/test.py projects/dsp-sumac/resnet50_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn50_4/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn50_4"
#   "python tools/test.py projects/dsp-sumac/resnet101_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn101_4/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn101_4"

#   "python tools/test.py projects/dsp-sumac/hrnet-w32_4xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/hrnet_5/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/hrnet_5"
#   "python tools/test.py projects/dsp-sumac/resnet50_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn50_5/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn50_5"
#   "python tools/test.py projects/dsp-sumac/resnet101_8xb32_dsp.py $(cat work_dirs/sumac_singlegpu/without_context/default/rn101_5/last_checkpoint) --work-dir work_dirs/sumac_singlegpu/without_context/default/rn101_5"
# )

# srun ${commands[$SLURM_ARRAY_TASK_ID]}
