_base_ = '../../msg3d/msg3d_pyskl_ntu60_xsub_hrnet/j.py'

data = dict(
    workers_per_gpu=0,
    persistent_workers=False,
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=0),
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=0),
)
load_from = 'checkpoints/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/j.pth'
work_dir = './work_dirs/modern/msg3d/ntu60_xsub_hrnet_joint'
