_base_ = 'mmpretrain::_base_/datasets/voc_bs16.py'

SNIFFYART_CLASSES = [
    'cooking', 'dancing', 'drinking', 'eating', 'holding the nose', 'painting', 'peeing',
    'playing music', 'praying', 'reading', 'sleeping', 'smoking', 'sniffing', 'textile work',
    'writing'
]
num_classes = len(SNIFFYART_CLASSES)

data_preprocessor = dict(
    num_classes = num_classes
)

train_dataloader = dict(
    dataset = dict(
        data_root='/hdd/datasets/sniffyart-extension/cls/VOC2012',
        split='train',
        metainfo = dict(
            classes=SNIFFYART_CLASSES
            )
    )
)
val_dataloader = dict(
    dataset = dict(
        data_root='/hdd/datasets/sniffyart-extension/cls/VOC2012',
        split='val',
        metainfo = dict(
            classes=SNIFFYART_CLASSES
        )
    )
)
test_dataloader = val_dataloader
