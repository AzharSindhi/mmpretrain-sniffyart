#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --job-name=sniffy_adho_test
#SBATCH --gres=gpu:2
#SBATCH --partition=a100,v100

DIR=/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/
cd $DIR

source venv/bin/activate
python tools/train.py projects/adho2024/without_context/resnet101_8xb32_dsp.py --work-dir work_dirs/adho/without/default/rn101_1
