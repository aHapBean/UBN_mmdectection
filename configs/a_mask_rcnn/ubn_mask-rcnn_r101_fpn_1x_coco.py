_base_ = './ubn_mask-rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='./checkpoints/r101_ubn.pth')))
