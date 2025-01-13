from mmengine.optim import AmpOptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from mmengine.visualization import LocalVisBackend, WandbVisBackend
from torch.optim import AdamW
from mmseg.visualization import SegLocalVisualizer
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook)
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import PolyLR, LinearLR, CosineAnnealingLR
from mmengine.runner.loops import IterBasedTrainLoop, TestLoop, ValLoop
from mmseg.engine import SegVisualizationHook
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import (RandomFlip, RandomResize, Resize, TestTimeAug)
from mmengine.dataset.sampler import DefaultSampler
from mmseg.datasets.transforms.formatting import PackSegInputs
from rsris import RefSegIoUMetric
from rsris.datasets.refdataset import RefSegDataset, LoadSegAnnotations
from rsris.models.models import RefSegEncoderDecoder, RefSegSamVisionEncoder, RefSegSiglipVisionModel, RefSegSiglipTextModel, RefSegSamPromptEncoder, RefSegSamMaskDecoder

default_scope = 'mmseg'
custom_imports = dict(imports=['rsris'], allow_failed_imports=False)

work_dir = f'./work_dirs/RSRefSeg-l'
data_root = f'/mnt/dataset/cky_data/RRSIS-D/images/rrsisd/JPEGImages'
batch_size = 1
max_epochs = 200
val_interval = 5

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type=LocalVisBackend),
    # dict(type=WandbVisBackend, init_kwargs=dict(project='RSRefSeg', group='RSRefSeg-b', name=work_dir.split('/')[-1]))
]
visualizer = dict(type=SegLocalVisualizer, vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

init_from = None
# init_from = dict(
#     type='Pretrained',
#     checkpoint='work_dirs/RSRefSeg-l/epoch_85.pth/mp_rank_00_model_states.pth'
# )


param_scheduler = [
    dict(type=LinearLR, start_factor=0.01, by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=max_epochs,
        by_epoch=True,
        begin=5,
        eta_min_ratio=0.001,
        end=max_epochs),
]


train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=True, interval=val_interval, max_keep_ckpts=5, save_last=True, greater_keys=['RefSeg/gIoU_1']),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook)
)


crop_size = (1024, 1024)
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadSegAnnotations),
    dict(type=Resize, scale=crop_size, keep_ratio=False),
    dict(type=PackSegInputs,
         meta_keys=('text',
                    'img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label')
        )
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=crop_size, keep_ratio=False),
    dict(type=LoadSegAnnotations),
    dict(type=PackSegInputs,
         meta_keys=('text',
                    'img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label')
         )
]

# dataset settings
dataset_type = RefSegDataset
num_workers = 8
persistent_workers = True

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='datainfo/rrsisd_train.jsonl',
        pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='datainfo/rrsisd_val.jsonl',
        pipeline=test_pipeline,
        test_mode=True
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(type=RefSegIoUMetric)
test_evaluator = val_evaluator


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],  # normalize the image in the model internally
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

model = dict(
    type=RefSegEncoderDecoder,
    data_preprocessor=data_preprocessor,
    down_spatial_times=2,
    with_dense_feat=True,
    init_cfg=init_from,
    lora_cfg=dict(
        backbone=dict(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['qkv', 'proj', 'lin1', 'lin2', 'neck.conv1', 'neck.conv2']
        ),
        clip_vision_encoder=dict(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['k_proj', 'v_proj', 'q_proj', 'out_proj']
        ),
        clip_text_encoder=dict(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            # target_modules='(^model\.encoder\.layers\.(1[4-9]|2[0-7])\.self_attn\.(k_proj|v_proj|q_proj|out_proj))|(^model\.head)'
            target_modules='(^model\.encoder\.layers\..*\.self_attn\.(k_proj|v_proj|q_proj|out_proj))|(^model\.head)'
        ),
    ),
    backbone=dict(
        type=RefSegSamVisionEncoder,
        model_name_or_path= 'KyanChen/sam-vit-large'
    ),
    clip_vision_encoder=dict(
        type=RefSegSiglipVisionModel,
        model_name_or_path='thomas/siglip-so400m-patch14-384',
    ),
    clip_text_encoder=dict(
        type=RefSegSiglipTextModel,
        model_name_or_path='thomas/siglip-so400m-patch14-384',
    ),
    sam_prompt_encoder=dict(
        type=RefSegSamPromptEncoder,
        model_name_or_path='KyanChen/sam-vit-large',
    ),
    sam_mask_decoder=dict(
        type=RefSegSamMaskDecoder,
        model_name_or_path='KyanChen/sam-vit-large',
    ),
    decode_head=dict(
        type='PseudoSegHead',
        num_classes=2,
        out_channels=1,
        threshold=0.5,
        align_corners=False,
        loss_decode=dict(type='mmseg.CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

base_lr = 0.0001
find_unused_parameters=True

#### AMP training config
# runner_type = 'Runner'
# optim_wrapper = dict(
#     type=AmpOptimWrapper,
#     dtype='bfloat16',  # float16
#     optimizer=dict(type=AdamW, lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
# )

### DeepSpeed training config
runner_type = 'FlexibleRunner'
strategy = dict(
    type='DeepSpeedStrategy',
    fp16=dict(
        enabled=True,
        auto_cast=False,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    inputs_to_half=['inputs'],
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        allgather_bucket_size=2e8,
        reduce_scatter=True,
        reduce_bucket_size='auto',
        overlap_comm=True,
        contiguous_gradients=True,
    ),
)
optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05
    )
)