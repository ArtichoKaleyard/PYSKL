_base_ = '../../posec3d/slowonly_r50_ntu60_xsub/joint.py'

data = dict(
    workers_per_gpu=0,
    persistent_workers=False,
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=0),
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=0),
)
load_from = 'checkpoints/posec3d/slowonly_r50_ntu60_xsub/joint.pth'
work_dir = './work_dirs/modern/posec3d/ntu60_xsub_joint'
