_base_ = [
    '../../../configs/_base_/models/hrnet/hrnet-w32.py', # model
    'dsp_cls.py', # dataset
    '../../../configs/_base_/schedules/imagenet_bs256_coslr.py',
    '../../../configs/_base_/default_runtime.py'
]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (4 GPUs) x (32 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)
feat_dim = 1024
out_dim = 512
# model_neck = dict(
#     type="NonLinearNeck",
#     in_channels = feat_dim,
#     hid_channels: int,
#     out_channels: int,
#     num_layers: int = 2,
# )

model = dict(
    type='TwoBranchModel',
    backbone=dict(init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmpose/'
        'pretrain_models/hrnet_w32-36af842e.pth')),
    use_context = True,
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