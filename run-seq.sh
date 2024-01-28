# bash ./tools/dist_train.sh ./configs/a_faster_rcnn/bn_faster-rcnn_r50_fpn_1x_coco.py 8

# dect   resnet101
# bash ./tools/dist_train.sh ./configs/a_faster_rcnn/ubn_faster-rcnn_r101_fpn_1x_coco.py 8

# bash ./tools/dist_train.sh ./configs/a_faster_rcnn/bn_faster-rcnn_r101_fpn_1x_coco.py 8


# seg    resnet50
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/bn_mask-rcnn_r50_fpn_1x_coco.py 8

bash ./tools/dist_train.sh ./configs/a_mask_rcnn/ubn_mask-rcnn_r50_fpn_1x_coco.py 8


# seg    resnet101
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/bn_mask-rcnn_r101_fpn_1x_coco.py 1

bash ./tools/dist_train.sh ./configs/a_mask_rcnn/ubn_mask-rcnn_r101_fpn_1x_coco.py 1

# ubn mask rcnn

# 如果有空余时间考虑跑 resnet101 的实验

# 1.27这是 tmux 00 53000 正在跑8卡 
# ubn faster rcnn r50 √
