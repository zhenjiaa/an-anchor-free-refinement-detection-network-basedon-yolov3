#!/bin/bash

LOG=log/train_720p-`date +%Y-%m-%d-%H-%M-%S`.log

python  train.py \
    --img 320 \
    --batch 32 \
    --epochs 300 \
    --data cfg/ccpd/box_720p.yaml \
    --cfg cfg/ccpd/refine_down4.yaml \
    --hyp data/hyp.scratch.yaml \
    --weight yolov3.pt \
    --project runs/train_720p \
    --device 1 2>&1 | tee $LOG