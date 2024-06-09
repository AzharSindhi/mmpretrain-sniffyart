_base_ = [
    '../../../configs/_base_/models/swin_transformer_v2/small_256.py',
    'deart_cls.py',
    '../../../configs/_base_/schedules/imagenet_bs32_adamw_swin.py',
    '../../../configs/_base_/default_runtime.py'
]

# neck = dict(type="NonLinearNeck",
#             in_channels = 768,
#             hid_channels = 512,
#             out_channels = 768,
#             num_layers = 3,
# )
model = dict(
    type='TwoBranchModel',
    backbone=dict(
        type='SwinTransformerV2',
        init_cfg=dict(type='Pretrained', checkpoint='/home/woody/iwi5/iwi5197h/mmpretrain-sniffyart/checkpoints/swinv2-small-w16_3rdparty_in1k-256px_20220803-b707d206.pth'),
        # window_size=[16, 16, 16, 8],
        img_size=224,
        pad_small_map=True,
    ),
    use_context = True,
    head=dict(
        num_classes=13,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, 
                  class_weight=[0.00412909, 0.05891123, 0.05766478, 0.01002574, 0.04495049,
                                0.00553355, 0.03603547, 0.01387911, 0.04623479, 0.03169092,
                                0.04498955, 0.53940594, 0.10654932]),                  topk=(1,2),
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