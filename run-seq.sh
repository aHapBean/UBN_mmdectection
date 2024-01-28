# bash ./tools/dist_train.sh ./configs/a_faster_rcnn/bn_faster-rcnn_r50_fpn_1x_coco.py 8



TODO:

# dect   resnet101    jxs2在跑这个
bash ./tools/dist_train.sh ./configs/a_faster_rcnn/ubn_faster-rcnn_r101_fpn_1x_coco.py 8

bash ./tools/dist_train.sh ./configs/a_faster_rcnn/bn_faster-rcnn_r101_fpn_1x_coco.py 8


# seg    resnet50    
# 53000 在跑这个
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/bn_mask-rcnn_r50_fpn_1x_coco.py 8
# pj2 跑这个
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/ubn_mask-rcnn_r50_fpn_1x_coco.py 8


# seg    resnet101

# 这个bn要不七卡跑完 TODO 
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/bn_mask-rcnn_r101_fpn_1x_coco.py 7
# TODO pj2 跑这个
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/ubn_mask-rcnn_r101_fpn_1x_coco.py 8

# ubn mask rcnn

# 貌似mask 也是 12 epochs 因为是 1x