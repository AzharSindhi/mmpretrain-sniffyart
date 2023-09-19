_base_ = [
    '../../configs/_base_/models/resnet101.py', # model
    'dsp_cls.py', # dataset
    '../../configs/_base_/schedules/imagenet_bs256.py',
    '../../configs/_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
    ),
    head=dict(
        num_classes=7,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, 
                  class_weight=[0.0736,0.1866,.0543,.4594,.0591,.0406,.1926]),
                  topk=(1,2),
    )
)
