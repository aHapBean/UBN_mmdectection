注意检查config文件的pth文件路径是否正确
ubn bound should be 0.15

1.
faster rcnn

# 53000 tmux 00 跑8卡
pj2 8卡
bash ./tools/dist_train.sh ./configs/a_faster_rcnn/bn_faster-rcnn_r50_fpn_1x_coco.py 8
 
ubn   
53000 8 卡
bash ./tools/dist_train.sh ./configs/a_faster_rcnn/ubn_faster-rcnn_r50_fpn_1x_coco.py 8



r101:
bash ./tools/dist_train.sh ./configs/a_faster_rcnn/bn_faster-rcnn_r101_fpn_1x_coco.py 8

bash ./tools/dist_train.sh ./configs/a_faster_rcnn/ubn_faster-rcnn_r101_fpn_1x_coco.py 8



目前看来这个用自己的pth是可能有效果的

停止训练用这个，不要ctrl+C，有bug
pkill -f "train.py"

8卡2bs是官方的标准配置
NOTE:No_pretrain !!!!

fuser -v /dev/nvidia1 | awk '{print $0}' |  xargs kill -9


2.
mask rcnn
TODO: 尝试UBN
no pretrain
53000 跑8卡 tmux 00(这机子是不是有问题)
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/bn_mask-rcnn_r50_fpn_1x_coco.py 8

bash ./tools/dist_train.sh ./configs/a_mask_rcnn/ubn_mask-rcnn_r50_fpn_1x_coco.py 8



r101: 
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/bn_mask-rcnn_r101_fpn_1x_coco.py 1

bash ./tools/dist_train.sh ./configs/a_mask_rcnn/ubn_mask-rcnn_r101_fpn_1x_coco.py 1









不同卡数跑出来的结果会有影响吗？？初步看loss好像2卡也差不多

研究一下如何指定模型路径以及名称
