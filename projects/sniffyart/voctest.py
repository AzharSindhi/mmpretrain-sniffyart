_base_ = 'mmpretrain::vgg/vgg16_8xb16_voc.py'

train_dataloader = dict(
    dataset = dict(
        data_root='/hdd/datasets/VOC/VOC2012',
        split='train'
    )
)
val_dataloader = dict(
    dataset = dict(
        data_root='/hdd/datasets/VOC/VOC2012',
        split='val'
    )
)
test_dataloader = val_dataloader