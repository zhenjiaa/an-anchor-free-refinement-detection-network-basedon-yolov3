#!/bin/bash

LOG=log/train_1080p_refine_ID1-`date +%Y-%m-%d-%H-%M-%S`.log

python  refine/train.py \
    --img 320 \
    --batch 32 \
    --epochs 300 \
    --data cfg/ccpd/box_1080p.yaml \
    --cfg cfg/ccpd/refine_down16.yaml \
    --hyp data/hyp.scratch.yaml \
    --weight yolov3.pt \
    --project runs/train_1080p_refine_ID1 \
    --device 3 2>&1 | tee $LOG