_base_ = [
    '../_base_/models/rmt/tiny_224.py', '../_base_/datasets/cub_bs8_448.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(drop_path_rate=0.1, img_size=448),
    head=dict(num_classes=200))

# dataset settings
dataset_type = 'CUB'

data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=448,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=1024,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='data/CUB_200_2011',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=1024,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='data/CUB_200_2011',
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# dataset settings
# train_dataloader = dict(
#     batch_size=1024,
#     dataset=dict(data_root='../../../../beegfs/ImageNet/ilsvrc12'),
# )

# val_dataloader = dict(
#     batch_size=1024,
#     dataset=dict(data_root='../../../../beegfs/ImageNet/ilsvrc12'),
# )

# test_dataloader = val_dataloader

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0), optimizer=dict(lr=0.001))
