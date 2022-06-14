
#!/bin/bash

LOG=log/train_fourpoint_ccpd_lq-`date +%Y-%m-%d-%H-%M-%S`.log

python  fourpoint/train.py \
    --img 320 \
    --batch 32 \
    --epochs 300 \
    --data cfg/ccpd/ccpd_train_data_with_test.yaml \
    --cfg cfg/ccpd/refine_down16.yaml \
    --hyp data/hyp.scratch.yaml \
    --weight yolov3.pt \
    --project runs/train_fourpoint_ccpd_lq \
    --device 1 2>&1 | tee $LOG
