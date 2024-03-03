_base_ = './csra_1xb16_sniffyart-448px_r101.py'

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

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