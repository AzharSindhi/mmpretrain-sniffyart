_base_ = './csra_r101.py'

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        arch='base',
        img_size=384,
        stage_cfgs=dict(block_cfgs=dict(window_size=12))
    ),
    head = dict(
        in_channels=1024
    )
)