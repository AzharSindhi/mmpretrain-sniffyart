_base_ = [
    '../../configs/_base_/models/hrnet/hrnet-w32.py', # model
    'dsp_cls.py', # dataset
    '../../configs/_base_/schedules/imagenet_bs256_coslr.py',
    '../../configs/_base_/default_runtime.py'
]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (4 GPUs) x (32 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)

model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint='../mmpose/work_dirs/hrnet_4/epoch_190.pth'),
    head=dict(
        num_classes=7,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, 
                  class_weight=[0.0736,0.1866,.0543,.4594,.0591,.0406,.1926]),
                  topk=(1,2),
    )
)