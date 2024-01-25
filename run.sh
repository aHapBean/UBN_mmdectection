python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cuda:0


bash ./tools/dist_train.sh ./configs/fast_rcnn/fast-rcnn_r50_fpn_1x_coco.py 1
bash ./tools/dist_train.sh ./configs/rpn/rpn_r50_fpn_1x_coco.py 1

./tools/dist_test.sh \
    configs/rpn/rpn_r50_fpn_1x_coco.py \
    checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth \
    8


# faster rcnn
# faster rcnn不要那个rpn吗？
bash ./tools/dist_train.sh ./configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py 1


faster rcnn
63000跑3卡bn tmux 00
bash ./tools/dist_train.sh ./configs/a_faster_rcnn/bn_faster-rcnn_r50_fpn_1x_coco.py 2

ubn
bash ./tools/dist_train.sh ./configs/a_faster_rcnn/ubn_faster-rcnn_r50_fpn_1x_coco.py 2

NOTE:No_pretrain !!!!


mask rcnn
TODO: 尝试UBN

64000 跑2卡 mask rcnn tmux 00
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py 2

不同卡数跑出来的结果会有影响吗？？初步看loss好像2卡也差不多
