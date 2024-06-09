_base_ = [
    '../../../configs/_base_/models/hrnet/hrnet-w32.py', # model
    'deart_cls.py', # dataset
    '../../../configs/_base_/schedules/imagenet_bs256_coslr.py',
    '../../../configs/_base_/default_runtime.py'
]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (4 GPUs) x (32 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)
# neck=[
#     dict(type='HRFuseScales', in_channels=(32, 64, 128, 256)),
#     dict(type="NonLinearNeck",in_channels = 2048,hid_channels = 1024,out_channels = 2048,num_layers = 3,)
# ]

model = dict(
    type='TwoBranchModel',
    backbone=dict(init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmpose/'
        'pretrain_models/hrnet_w32-36af842e.pth')),
    use_context = True,
    head=dict(
        num_classes=13,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, 
                  class_weight=[0.00412909, 0.05891123, 0.05766478, 0.01002574, 0.04495049,
                                0.00553355, 0.03603547, 0.01387911, 0.04623479, 0.03169092,
                                0.04498955, 0.53940594, 0.10654932]),
                  topk=(1,2),
    )
)

default_hooks = dict(
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',    # svae the best, auto select the `Accuracy` to the first  metric in val_evalutor
        interval=20,
        max_keep_ckpts=1,  # only save the  latest 3 ckpts
        rule='greater'            # the greater the metric, the better the ckpt will be    
)
)
# Training settings
# train_cfg = dict(
#     max_epochs=3,
# )