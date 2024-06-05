#!/usr/bin/env bash

# default neck: Global average pooling on the final feature map
# without context
tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/hrnet_1
tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn50_1
tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn101_1

tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/hrnet_2
tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn50_2
tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn101_2

tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/hrnet_3
tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn50_3
tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn101_3

tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/hrnet_4
tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn50_4
tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn101_4

tools/dist_train.sh projects/adho2024/without_context/hrnet-w32_4xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/hrnet_5
tools/dist_train.sh projects/adho2024/without_context/resnet50_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn50_5
tools/dist_train.sh projects/adho2024/without_context/resnet101_8xb32_dsp.py 2 --work-dir work_dirs/adho/without/default/rn101_5
