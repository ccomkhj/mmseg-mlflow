norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1, 1])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(96, 96), stride=(72, 72)))
dataset_type = 'LeafDataset'
data_root = '/home/huijo/Product/mmseg3/data'
img_norm_cfg = dict(
    mean=[112.6335, 130.305, 81.039],
    std=[59.415, 60.129, 57.2985],
    to_rgb=True)
img_scale = None
crop_size = (96, 96)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=None, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(96, 96), cat_max_ratio=0.5),
    dict(
        type='RandomRotate',
        prob=0.7,
        degree=90,
        pad_val=0,
        seg_pad_val=0,
        center=None,
        auto_bound=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[112.6335, 130.305, 81.039],
        std=[59.415, 60.129, 57.2985],
        to_rgb=True),
    dict(type='Pad', size=(96, 96), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[112.6335, 130.305, 81.039],
                std=[59.415, 60.129, 57.2985],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type='LeafDataset',
            data_root='/home/huijo/Product/mmseg3/data/20221031_combined',
            img_dir='train',
            ann_dir='ann',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=None, ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop', crop_size=(96, 96),
                    cat_max_ratio=0.5),
                dict(
                    type='RandomRotate',
                    prob=0.7,
                    degree=90,
                    pad_val=0,
                    seg_pad_val=0,
                    center=None,
                    auto_bound=False),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[112.6335, 130.305, 81.039],
                    std=[59.415, 60.129, 57.2985],
                    to_rgb=True),
                dict(type='Pad', size=(96, 96), pad_val=0, seg_pad_val=0),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='LeafDataset',
        data_root='/home/huijo/Product/mmseg3/data/20221031_combined',
        img_dir='val',
        ann_dir='ann',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[112.6335, 130.305, 81.039],
                        std=[59.415, 60.129, 57.2985],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='LeafDataset',
        data_root='/home/huijo/Product/mmseg3/data/20221031_combined',
        img_dir='val',
        ann_dir='ann',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[112.6335, 130.305, 81.039],
                        std=[59.415, 60.129, 57.2985],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=100000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
checkpoint_config = dict(by_epoch=False, interval=100000)
evaluation = dict(interval=20000, metric='mIoU', pre_eval=False)

gpu_ids = [0]
auto_resume = False

log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='MlflowLoggerHook', 
            exp_name='test1', 
            params={
                'crop_size': crop_size,
                'optimizer.type': optimizer.get('type'),
                'optimizer.lr': optimizer.get('lr'),
                'optimizer.momentum': optimizer.get('momentum'),
                'optimizer.weight_decay': optimizer.get('weight_decay'),
                'lr_config.policy': lr_config.get('policy'),
                'lr_config.power': lr_config.get('power'),
                'lr_config.min_lr': lr_config.get('min_lr'),
                'lr_config.policy': lr_config.get('policy'),
                'runner.type': runner.get('type'),
                'runner.max_iters': runner.get('max_iters'),
                'model.type': model.get('type'),
                'backbone.type': model.get('backbone').get('type'),
                'img_norm_cfg.mean': img_norm_cfg.get('mean'),
                'img_norm_cfg.std': img_norm_cfg.get('std'),
                'samples_per_gpu': data.get('samples_per_gpu'),
                'aug_seq_train': [i.get('type') for i in train_pipeline],
                'aug_seq_val': [i.get('type') for i in test_pipeline]
            },
            log_model=True, 
            by_epoch=False
            )
            ])