_base_ = './bn_faster-rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='/home/zhangxiangdong/mmdetection/checkpoints/r101_bn.pth')))
