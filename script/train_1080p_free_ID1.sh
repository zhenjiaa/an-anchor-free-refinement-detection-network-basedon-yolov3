#!/bin/bash

LOG=log/train_free_yolo_1080p-`date +%Y-%m-%d-%H-%M-%S`.log

python  free_yolo/train.py \
    --img 320 \
    --batch 32 \
    --epochs 300 \
    --data cfg/ccpd/box_1080p.yaml \
    --cfg free_yolo/free-yolov3.yaml \
    --hyp data/hyp.scratch.yaml \
    --weight yolov3.pt \
    --cache-images \
    --project runs/train_free_yolo_1080p \
    --device 3 2>&1 | tee $LOG