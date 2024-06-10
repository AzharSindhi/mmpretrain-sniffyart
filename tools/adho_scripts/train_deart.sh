#!/usr/bin/env bash

#SBATCH --time=09:00:00
#SBATCH --job-name=train_adho_deart
#SBATCH --gres=gpu:3
#SBATCH --partition=a100,v100
#SBATCH --array=0-5  # Adjust based on the number of experiments
#SBATCH --output=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_out.txt
#SBATCH --error=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/slurm_logs/%x_%j_err.txt

DIR=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/
cd $DIR

source venv/bin/activate

# Number of epochs
NUM_EPOCHS=100  # Set this to the desired number of epochs
BASE_WORK_DIR=work_dirs/adho_corrected_deart

# Define the commands
commands=(
  # SwinV2 with context
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/swinv2_1 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/swinv2_2 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/swinv2_3 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/swinv2_4 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/swinv2_5 --cfg-options runner.max_epochs=$NUM_EPOCHS"

  # SwinV2 without context
  # "tools/dist_train.sh projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/swinv2_1 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/swinv2_2 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/swinv2_3 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/swinv2_4 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/swinv2_5 --cfg-options runner.max_epochs=$NUM_EPOCHS"

  # ResNet 50 with context
  "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn50_1 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn50_2 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn50_3 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn50_4 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn50_5 --cfg-options runner.max_epochs=$NUM_EPOCHS"

  # ResNet 50 without context
  "tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn50_1 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn50_2 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn50_3 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn50_4 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn50_5 --cfg-options runner.max_epochs=$NUM_EPOCHS"

  # ResNet 101 with context
  "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn101_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn101_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn101_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn101_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/rn101_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"

  # ResNet 101 without context
  "tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn101_1 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn101_2 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn101_3 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn101_4 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/rn101_5 --cfg-options runner.max_epochs=$NUM_EPOCHS"

  # HRNet with context
  "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/hrnet_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/hrnet_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/hrnet_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/hrnet_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context/default/hrnet_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True"

  # HRNet without context
  "tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/hrnet_1 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/hrnet_2 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/hrnet_3 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/hrnet_4 --cfg-options runner.max_epochs=$NUM_EPOCHS"
  # "tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/without_context/default/hrnet_5 --cfg-options runner.max_epochs=$NUM_EPOCHS"

  # # SwinV2 with random context prob=1.0
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/swinv2_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/swinv2_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/swinv2_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/swinv2_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/swinv2_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"

  # # ResNet 50 with random context prob=1.0
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn50_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn50_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn50_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn50_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn50_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"

  # # ResNet 101 with random context prob=1.0
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn101_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn101_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn101_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn101_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/rn101_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"

  # # HRNet with random context prob=1.0
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/hrnet_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/hrnet_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/hrnet_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/hrnet_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context1.0/hrnet_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=1.0"

  # # SwinV2 with random context prob=0.15
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/swinv2_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/swinv2_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/swinv2_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/swinv2_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/swinv2_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"

  # # ResNet 50 with random context prob=0.15
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn50_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn50_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn50_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn50_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn50_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"

  # # ResNet 101 with random context prob=0.15
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn101_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn101_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn101_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn101_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/rn101_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"

  # # HRNet with random context prob=0.15
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/hrnet_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/hrnet_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/hrnet_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/hrnet_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py 3 --work-dir work_dirs/adho_corrected_deart/with_context_ablations/default/random_context0.15/hrnet_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.random_context=True train_dataloader.dataset.random_context_prob=0.15"

  # # SwinV2 with mask context box
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/swinv2-small-w16_16xb64_in1k-256px_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/swinv2_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"

  # # ResNet 50 with mask context box using deArt pretrained weights
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet50_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn50_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"

  # # ResNet 101 with mask context box using deArt pretrained weights
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/resnet101_8xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/rn101_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"

  # # HRNet with mask context box using deArt pretrained weights
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_1 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_2 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_3 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_4 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"
  # "tools/dist_train.sh projects/adho2024/with_context/hrnet-w32_4xb32_dsp_deart.py --work-dir $BASE_WORK_DIR/with_context_ablations/default/mask_context_box/hrnet_5 --cfg-options runner.max_epochs=$NUM_EPOCHS train_dataloader.dataset.mask_context_box=True"

)

# Execute the commands
srun ${commands[$SLURM_ARRAY_TASK_ID]}
