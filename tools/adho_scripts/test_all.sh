#!/usr/bin/env bash

#SBATCH --time=00:20:00
#SBATCH --job-name=test_adho
#SBATCH --gres=gpu:1
#SBATCH --partition=a100,v100
#SBATCH --array=0-1  # Adjust based on the number of experiments
#SBATCH --output=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_err.txt

DIR=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/
cd $DIR

source venv/bin/activate

# Number of epochs
BASE_WORK_DIR=work_dirs/adho_all_final
# Test commands
commands=(
  # SwinV2 with context
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context/default/swinv2_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/swinv2_1"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context/default/swinv2_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/swinv2_2"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context/default/swinv2_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/swinv2_3"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context/default/swinv2_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/swinv2_4"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context/default/swinv2_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/swinv2_5"

  # SwinV2 without context
  "python tools/test.py projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/without_context/default/swinv2_1/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/swinv2_1"
  "python tools/test.py projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/without_context/default/swinv2_2/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/swinv2_2"
  "python tools/test.py projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/without_context/default/swinv2_3/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/swinv2_3"
  "python tools/test.py projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/without_context/default/swinv2_4/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/swinv2_4"
  "python tools/test.py projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/without_context/default/swinv2_5/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/swinv2_5"

  # ResNet 50 with context
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn50_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn50_1"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn50_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn50_2"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn50_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn50_3"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn50_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn50_4"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn50_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn50_5"

  # ResNet 50 without context
  "python tools/test.py projects/adho2024/without_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn50_1/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn50_1"
  "python tools/test.py projects/adho2024/without_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn50_2/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn50_2"
  "python tools/test.py projects/adho2024/without_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn50_3/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn50_3"
  "python tools/test.py projects/adho2024/without_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn50_4/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn50_4"
  "python tools/test.py projects/adho2024/without_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn50_5/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn50_5"

  # ResNet 101 with context
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn101_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn101_1"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn101_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn101_2"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn101_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn101_3"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn101_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn101_4"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/rn101_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/rn101_5"

  # ResNet 101 without context
  "python tools/test.py projects/adho2024/without_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn101_1/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn101_1"
  "python tools/test.py projects/adho2024/without_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn101_2/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn101_2"
  "python tools/test.py projects/adho2024/without_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn101_3/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn101_3"
  "python tools/test.py projects/adho2024/without_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn101_4/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn101_4"
  "python tools/test.py projects/adho2024/without_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/rn101_5/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/rn101_5"

  # HRNet with context
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/hrnet_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/hrnet_1"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/hrnet_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/hrnet_2"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/hrnet_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/hrnet_3"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/hrnet_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/hrnet_4"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context/default/hrnet_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/hrnet_5"

  # HRNet without context
  "python tools/test.py projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/hrnet_1/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/hrnet_1"
  "python tools/test.py projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/hrnet_2/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/hrnet_2"
  "python tools/test.py projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/hrnet_3/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/hrnet_3"
  "python tools/test.py projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/hrnet_4/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/hrnet_4"
  "python tools/test.py projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/without_context/default/hrnet_5/last_checkpoint) --work-dir $BASE_WORK_DIR/without_context/default/hrnet_5"

  # SwinV2 with random context prob=1.0
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_1"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_2"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_3"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_4"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/swinv2_5"

  # ResNet 50 with random context prob=1.0
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_1"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_2"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_3"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_4"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn50_5"

  # ResNet 101 with random context prob=1.0
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_1"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_2"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_3"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_4"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/rn101_5"

  # HRNet with random context prob=1.0
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_1"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_2"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_3"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_4"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context1.0/hrnet_5"

  # SwinV2 with random context prob=0.15
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_1"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_2"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_3"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_4"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/swinv2_5"

  # ResNet 50 with random context prob=0.15
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_1"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_2"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_3"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_4"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn50_5"

  # ResNet 101 with random context prob=0.15
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_1"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_2"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_3"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_4"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/rn101_5"

  # HRNet with random context prob=0.15
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_1"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_2"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_3"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_4"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/random_context0.15/hrnet_5"
  
  # SwinV2 with mask context box
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_1"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_2"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_3"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_4"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_5"

  # ResNet 50 with mask context box
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_1"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_2"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_3"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_4"
  "python tools/test.py projects/adho2024/with_context/resnet50_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_5"

  # ResNet 101 with mask context box
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_1"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_2"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_3"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_4"
  "python tools/test.py projects/adho2024/with_context/resnet101_8xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_5"

  # HRNet with mask context box
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_1/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_1"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_2/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_2"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_3/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_3"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_4"
  "python tools/test.py projects/adho2024/with_context/hrnet-w32_4xb32_dsp.py $(cat $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_5"

)

new_commands=(
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context/default/swinv2_4/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/swinv2_4"
  "python tools/test.py projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px.py $(cat $BASE_WORK_DIR/with_context/default/swinv2_5/last_checkpoint) --work-dir $BASE_WORK_DIR/with_context/default/swinv2_5"

)
srun ${new_commands[$SLURM_ARRAY_TASK_ID]}