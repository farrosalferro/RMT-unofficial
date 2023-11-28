_base_ = [
    '../_base_/models/rmt/tiny_224.py', '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

model = dict(backbone=dict(img_size=512), head=dict(num_classes=10))

# dataset setting
train_pipeline = [
    dict(type='RandomResizedCrop', scale=512),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=512),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline), num_workers=1)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline), num_workers=1)
test_dataloader = dict(dataset=dict(pipeline=test_pipeline), num_workers=1)

# schedule setting
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
