_base_ = [
    '../../../configs/_base_/models/resnet50.py', # model
    'dsp_cls.py', # dataset
    '../../../configs/_base_/schedules/imagenet_bs256.py',
    '../../../configs/_base_/default_runtime.py'
]

model = dict(
    type='TwoBranchModel',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    head=dict(
        num_classes=7,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, 
                  class_weight=[0.0736,0.1866,.0543,.4594,.0591,.0406,.1926]),
                  topk=(1,2),
    )
)

default_hooks = dict(
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',    # svae the best, auto select the `Accuracy` to the first  metric in val_evalutor
        interval=1,
        max_keep_ckpts=3,  # only save the  latest 3 ckpts
        rule='greater'            # the greater the metric, the better the ckpt will be    
)
)

# Training settings
train_cfg = dict(
    max_epochs=3,
)