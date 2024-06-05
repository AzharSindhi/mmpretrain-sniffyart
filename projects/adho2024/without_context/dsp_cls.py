# dataset settings
dataset_type = 'ImagePairDataset'
# images directory
images_dir = "/net/cluster/azhar/datasets/gestuers_keypoints_cmac/smell-gesture-recognition/images"
# train json path
train_anns_path = "/net/cluster/azhar/mywork/thesis-gestures-recognition/data/gestures/annotations_train.json"
# validation json path
val_anns_path = "/net/cluster/azhar/mywork/thesis-gestures-recognition/data/gestures/annotations_valid.json"
# test json path
test_anns_path = "/net/cluster/azhar/mywork/thesis-gestures-recognition/data/gestures/annotations_test.json"

data_preprocessor = dict(
    num_classes=7,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=False,
)

use_context = False

train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        ann_file = train_anns_path,
        data_root=images_dir,
        use_context = use_context,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        ann_file=val_anns_path,
        data_root=images_dir,
        use_context = use_context,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        ann_file=test_anns_path,
        data_root=images_dir,
        use_context = use_context,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = [
    dict(type='Accuracy', topk=(1,2)),
    dict(type='SingleLabelMetric'),
    dict(type='ConfusionMatrix'),
]


test_evaluator = val_evaluator