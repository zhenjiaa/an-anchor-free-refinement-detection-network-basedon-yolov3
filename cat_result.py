#!/usr/bin/env python

import argparse
import time
import os
import cv2
import torch
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess(im, imsz):
    im = letterbox(im, new_shape = imsz)[0]
    im = im[:, :, ::-1].transpose(2, 0 ,1)
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255

    return im

def preprocess_yuv(im, imsz):


    h_,w_,_ = im.shape
    if h_%4!=0:
        h_ = (int(h_/4)+1)*4
    if w_%2!=0:
        w_ = (int(w_/2)+1)*2

    im =cv2.resize(im,(w_,h_))
    im_yuv = cv2.cvtColor(im,cv2.COLOR_BGR2YUV_I420)
    h_,w_ = im_yuv.shape
    h_ = int(h_/1.5)
    y = im_yuv[0:h_,0:w_]
    u = im_yuv[h_:int(h_*1.25),0:w_].reshape(int(h_/2),int(w_/2))
    v = im_yuv[int(h_*1.25):int(h_*1.5),0:w_].reshape(int(h_/2),int(w_/2))
    y = y[::2,::2]
    im = np.stack((y, u, v), axis=2)


    im = letterbox(im, new_shape = imsz)[0]
    # print(im.shape)

    # im_yuv = cv2.cvtColor(im,cv2.COLOR_BGR2YUV_I420)

    # h_,w_ = im_yuv.shape
    # h_ = int(h_/1.5)
    # y = im_yuv[0:h_,0:w_]
    # u = im_yuv[h_:int(h_*1.25),0:w_].reshape(int(h_/2),int(w_/2))
    # v = im_yuv[int(h_*1.25):int(h_*1.5),0:w_].reshape(int(h_/2),int(w_/2))
    # y = cv2.resize(y,(int(w_/2),int(h_/2)))
    # im = np.stack((y, u, v), axis=2)
    # print(im.shape)
    im = im.transpose(2, 0 ,1)
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255
    return im

def detect(save_img=False):
    weights = opt.weights
    src = opt.src
    dst = opt.dst
    imsz = opt.img_size
    save_txt = opt.save_txt

    conf_thresh = opt.conf_thres
    iou_thresh = opt.iou_thres
    classes = opt.classes
    agnostic = opt.agnostic_nms

    # Initialize
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()

    f = open(os.path.join(dst, 'pred.txt'), 'w')
    for index, imname in enumerate(os.listdir(src)):
        print(imname)
        impath = os.path.join(src, imname)
        im0 = cv2.imread(impath)
        im1 = im0.copy()
        if not opt.yuv:
            im2 = preprocess(im0, imsz)
        else:
            im2 = preprocess_yuv(im0, imsz)

        input_tensor = torch.from_numpy(im2).to(device)
        input_tensor = input_tensor.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(input_tensor, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes, agnostic)
        t2 = time_synchronized()

        log = '%d:%s: %.4fs' % (index, impath, (t2 - t1))
        print (log)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            save_img = os.path.join(dst, imname)
            gn = torch.tensor(im1.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(input_tensor.shape[2:], det[:, :4], im1.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        cls = int(cls)
                        conf = float(conf)
                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        obj_info = '%s %d %.4f %d %d %d %d\n'% (imname, cls,conf,x1,y1,x2,y2)
                        f.write(obj_info)
                    
                    if save_img:  # Add bbox to image
                        if conf > 0.45:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im1, label=label, color=colors[int(cls)], line_thickness=3)

            if save_img:
                # cv2.imwrite(save_img, np.hstack((im1,im0)))
                cv2.imwrite(save_img, im1)
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/data1/diedai/7_0924/free-yolov3/runs/train_720p/exp3/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--src', type=str, default='/data0/TILT/720p/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--dst', type=str, default='./output_720p_yolov3', help='detect result')
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', default = True,action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    if not os.path.exists(opt.dst):
        os.mkdir(opt.dst)

    with torch.no_grad():
        detect()

