_base_ = [
    'mmpretrain::_base_/datasets/voc_bs16.py',
    'mmpretrain::_base_/default_runtime.py'
]

data_preprocessor = dict(
    num_classes=20,
    to_onehot=True,
    to_rgb=True
)

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

num_classes = 20

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict( type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=None,
    head=dict(
        type='CSRAClsHead',
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        num_classes=num_classes,
        in_channels=2048,
        num_heads=1,
        lam=0.1
    ))


train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0),
    # update the final linear by 10 times learning rate.
    paramwise_cfg=dict(custom_keys={'.backbone.classifier': dict(lr_mult=10)}),
)

# learning policy
param_scheduler = dict(type='StepLR', by_epoch=True, step_size=20, gamma=0.1)