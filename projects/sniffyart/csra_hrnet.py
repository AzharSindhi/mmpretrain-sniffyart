_base_ = './csra_r101.py'

checkpoint = '/home/woody/iwi5/iwi5093h/models/hrnet_w32-36af842e.pth'

model = dict(
   backbone=dict(
        _delete_=True,
        type='HRNet',
        arch='w32',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint)
    ),
    neck=dict(
        _delete_=True,
        type='HRFuseScales', in_channels=(32,64,128,256))
)
