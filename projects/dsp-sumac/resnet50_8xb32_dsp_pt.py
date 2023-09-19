_base_ = [
    '../../configs/_base_/models/resnet50.py', # model
    'dsp_cls.py', # dataset
    '../../configs/_base_/schedules/imagenet_bs256.py',
    '../../configs/_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='../mmdetection/work_dirs/frcnn_rn50_3/epoch_50.pth')
    ),
    head=dict(
        num_classes=7,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, 
                  class_weight=[0.0736,0.1866,.0543,.4594,.0591,.0406,.1926]),
                  topk=(1,2),
    )
)
