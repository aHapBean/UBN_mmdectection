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

换成了我们的pth文件 ！！！ 注意更改config文件(enable pretrain)  注意更改模型文件！！！！跑前检查（同时注意load信息）
53000 tmux 00 跑8卡
bash ./tools/dist_train.sh ./configs/a_faster_rcnn/bn_faster-rcnn_r50_fpn_1x_coco.py 8
 
ubn  63000跑跑看 tmux 00  用的我们的pth文件   注意更改模型文件！！！！跑前检查（同时注意load信息）
bash ./tools/dist_train.sh ./configs/a_faster_rcnn/ubn_faster-rcnn_r50_fpn_1x_coco.py 3

1.26TODO
每次跑之前注意要检查是哪一个模型文件，ubn还是bn！！！！！
重点观察这俩的性能表现是否正常（no pretrain的性能太过离谱了，太低了）

杀进程：
pkill -f "train.py"

8卡2bs是官方的标准配置
NOTE:No_pretrain !!!!

fuser -v /dev/nvidia1 | awk '{print $0}' |  xargs kill -9



mask rcnn
TODO: 尝试UBN
no pretrain
53000 跑8卡 tmux 00(这机子是不是有问题)
bash ./tools/dist_train.sh ./configs/a_mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py 8

不同卡数跑出来的结果会有影响吗？？初步看loss好像2卡也差不多
