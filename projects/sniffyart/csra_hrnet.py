_base_ = './csra_r101.py'

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth'

model = dict(
   backbone=dict(
        _delete_=True,
        type='HRNet',
        arch='w32',
        init_cfg=dict(
            type='Pretrained',
            #checkpoint='/home/woody/iwi5/iwiI5093h/models/hrnet_w32-36af842e.pth')
            checkpoint='https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w32_3rdparty_8xb32_in1k_20220120-c394f1ab.pth')
    ),
    neck=dict(
        _delete_=True,
        type='HRFuseScales', in_channels=(32,64,128,256))
)
