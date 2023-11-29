_base_ = [
    '../_base_/models/rmt/ex_384.py', '../_base_/datasets/cub_bs8_448.py',
    '../_base_/schedules/cub_bs64.py', '../_base_/default_runtime.py'
]

model = dict(backbone=dict(img_size=448), head=dict(num_classes=200))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=5e-6,
        weight_decay=0.0005,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
)

train_dataloader = dict(num_workers=1, batch_size=3)
val_dataloader = dict(num_workers=1, batch_size=3)
test_dataloader = dict(num_workers=1, batch_size=3)

default_hooks = dict(
    # log every 20 intervals
    logger=dict(type='LoggerHook', interval=20),
    # save last three checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
