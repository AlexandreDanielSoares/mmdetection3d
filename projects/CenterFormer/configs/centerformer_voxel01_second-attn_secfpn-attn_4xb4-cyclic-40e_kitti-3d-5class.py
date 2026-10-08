_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.CenterFormer.centerformer'], allow_failed_imports=False)

# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
# voxel_size = [0.1, 0.1, 0.15] old value for waymo dataset
# The new value generates a comparable number of voxels for the Kitti dataset as the original value for waymo dataset
# The current value will need to be reduced because I don't have enough GPU memory for training
# TODO: Try train the model and adjust this value acordingly also consider the grid_size and sparse_shape
# I tried to train the model and I don't have sufficient VRAM so need to increase voxel size 
#voxel_size = [0.05, 0.05, 0.1]
voxel_size = [0.2, 0.2, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
data_root = 'data/kitti/'
class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']
tasks = [dict(num_class=5, class_names=['car', 'van', 'truck', 'pedestrian', 'cyclist'])]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

model = dict(
    type='CenterFormer',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1))),
    voxel_encoder=dict(
        type='DynamicSimpleVFE',
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        # TODO: Check this value
        #sparse_shape=[41, 1600, 1408],
        sparse_shape=[41, 400, 352],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, [0, 1, 1]), (1, 1)),
        block_type='basicblock'),
    backbone=dict(
        type='DeformableDecoderRPN',
        layer_nums=[5, 5, 1],
        ds_num_filters=[256, 256, 128],
        num_input_features=256,
        tasks=tasks,
        classes=len(class_names),
        use_gt_training=True,
        corner=True,
        assign_label_window_size=1,
        obj_num=500,
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        transformer_config=dict(
            depth=2,
            n_heads=6,
            dim_single_head=64,
            dim_ffn=256,
            dropout=0.3,
            out_attn=False,
            n_points=15,
        ),
    ),
    bbox_head=dict(
        type='CenterFormerBboxHead',
        in_channels=256,
        tasks=tasks,
        dataset='kitti',
        weight=2,
        corner_loss=True,
        iou_loss=True,
        assign_label_window_size=1,
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={
            'reg': (2, 2),
            'height': (1, 2),
            'dim': (3, 2),
            'rot': (2, 2),
            'iou': (1, 2)
        },  # (output_channel, num_conv)
        iou_factor=[1, 1, 1, 1, 4]
    ),
    train_cfg=dict(
        # TODO: Check this value
        #grid_size=[1408, 1600, 40],
        grid_size=[352, 400, 40],
        voxel_size=voxel_size,
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        point_cloud_range=point_cloud_range,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    test_cfg=dict(
        post_center_limit_range=[0, -40, -3, 70.4, 40, 1],
        nms=dict(
            use_rotate_nms=False,
            use_multi_class_nms=True,
            nms_pre_max_size=[1600, 1600, 1600, 1600, 800],
            nms_post_max_size=[200, 200, 200, 200, 100],
            nms_iou_threshold=[0.8, 0.55, 0.55, 0.55, 0.55],
        ),
        score_threshold=0.1,
        pc_range=[0, -40],
        out_size_factor=4,
        voxel_size=[0.2, 0.2],
        obj_num=1000,
    ))

data_root = 'data/kitti/'
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Van=5, Truck=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Van=10, Truck=10, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity=True,
        backend_args=backend_args),
    # Add this if using `MultiFrameDeformableDecoderRPN`
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     load_dim=4,
    #     use_dim=4,
    #     pad_empty_sweeps=True,
    #     remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.5, 0.5, 0]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity=True,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(
        type='Pack3DDetInputs',
        keys=['points'])
]

dataset_type = 'KittiDataset'
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='KittiMetric', 
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='Dissertação Mestrado',                  # Change to your W&B project
            name='CenterFormer_Training_Kitti_40epochs_test6',     # Optional: experiment name
            #id='CenterFormer_KITTI_5class__test2',                  # set to previous run's id to resume the same run
            resume='allow',                                  # 'allow'|'must'|'never' (use 'allow' to resume if exists)
            save_code=True                                   # Optional: save code to W&B                             
            ))
]

visualizer = dict(
    type='Det3DLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')

# For waymo dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
lr = 3e-4
#lr = 0.0054
# The optimizer follows the setting in SECOND.Pytorch, but here we use
# the official AdamW optimizer implemented by PyTorch.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 16 epochs, learning rate increases from 0 to lr * 10
    # during the next 24 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=16,
        eta_min=lr * 10,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=lr * 1e-4,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 16 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 24 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=16,
        eta_min=0.85 / 0.95,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=24,
        eta_min=1,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (1 GPUs) x (4 samples per GPU).
# TODO: Check this value if I activate it 
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=30)]
