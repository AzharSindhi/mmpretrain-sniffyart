_base_ = './csra_r101.py'

checkpoint = '/home/woody/iwi5/iwi5093h/models/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

model = dict(
    backbone = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone'
        )
    )
)