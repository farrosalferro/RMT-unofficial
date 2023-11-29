_base_ = [
    '../_base_/models/rmt/tiny_224.py', '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py', '../_base_/default_runtime.py'
]

model = dict(backbone=dict(drop_path_rate=0.1),
             train_cfg=dict(augments=[
                 dict(type='RandomErasing', erase_prob=0.5)
             ]))

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0),
                     optimizer=dict(lr=0.001))
