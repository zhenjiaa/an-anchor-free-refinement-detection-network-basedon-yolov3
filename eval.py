#!/usr/bin/env python
# coding=utf-8

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2

class_map = {0:'person', 1:'cat', 2:'dog'}

def load_xml(labels_dir):
    labels = {}
    prefix = [os.path.splitext(x)[0] for x in os.listdir(labels_dir)]
    for index in prefix:
        xml_path = os.path.join(labels_dir, index + '.xml')
        tree = ET.parse(xml_path)
        objects = []
        for obj in tree.findall('object'):
            cls = obj.find('name').text

            cls = 0

            # if cls == 'bodyA':
            # if cls == 'bodyA' or cls == 'bodyB':
            # if cls != 'bodyA' and cls != 'bodyB' and cls != 'child':
            #     cls = 0
            # else:
            #     continue
        
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            objects.append((cls, xmin, ymin, xmax, ymax))
        labels[index] = objects
    return labels

def load_labels(labels_dir):
    labels = {}
    prefix = [os.path.splitext(x)[0] for x in os.listdir(labels_dir)]
    for index in prefix:
        if 'fourpoint' in index:
            continue
        label_path = os.path.join(labels_dir, index + '.txt')
        img = '/data0/TILT/720p/images/'+index+'.jpg'
        img = cv2.imread(img)
        h,w,_ =img.shape 
        # print(w,h)
        objects = []
        for line in open(label_path):
            line = line.strip()
            spline = line.split()
            # print(spline)
            cls = int(spline[0])
            # xmin = int(spline[3])
            # ymin = int(spline[5])
            # xmax = int(spline[4])
            # ymax = int(spline[6])
            x_c = int(float(spline[1])*w)
            y_c = int(float(spline[2])*h)
            w = int(float(spline[3])*w)
            h = int(float(spline[4])*h)
            xmin = x_c - w/2
            xmax = x_c + w/2
            ymin = y_c - h/2
            ymax = y_c + h/2
            objects.append((cls, xmin, ymin, xmax, ymax))
        labels[index] = objects
    return labels

def load_preds(pred_file):
    preds = {}
    for line in open(pred_file).readlines():
        imname, cls, prob, xmin, ymin, xmax, ymax = line.strip().split()
        cls = int(cls)
        prob = float(prob)
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        index = os.path.splitext(imname)[0]
        if cls not in preds:
            preds[cls] = []
        preds[cls].append([index, prob, xmin, ymin, xmax, ymax])
    return preds

def parse_preds(preds):
    res = {}
    for cls, class_preds in preds.items():
        preds_by_image = {}
        for obj in class_preds:
            prefix, prob = obj[0:2]
            if prefix not in preds_by_image:
                preds_by_image[prefix] = []
            preds_by_image[prefix].append(prob)
        res[cls] = []
        for prefix, prob in preds_by_image.items():
            res[cls].append(max(prob))
    return res

def FPR(results, thresh, total):
    fp = 0
    for prob in results:
        if prob > thresh and prob < 1.00:
            fp += 1
    return fp / total

def eval_ap(rec, prec):
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11
    return ap

def eval(labes, preds, iou_thresh, conf_thresh):
    avg_rec = []
    avg_prec = []
    mAP = []

    rec_dict = {}
    prec_dict = {}

    for cls, class_preds in preds.items():
        sorted_preds = sorted(class_preds, key = lambda x:x[1], reverse = True)

        class_labels = {}
        npos = 0
        for index, objects in labels.items():
            bbox = np.array([obj[1:] for obj in objects if obj[0] == cls])
            npos += len(bbox)
            det = [False] * len(bbox)
            class_labels[index] = {'bbox': bbox, 'det': det}

        nd = len(sorted_preds)
        tp = 0
        fp = 0
        TP = np.zeros(nd)
        FP = np.zeros(nd)
        for i, pred in enumerate(sorted_preds):
            index = pred[0]
            prob = pred[1]
            # if prob < conf_thresh:
                # continue

            bb = np.array(pred[2:6]).astype(float)
            bbgt = class_labels[index]['bbox'].astype(float) 
            
            iou_max = -np.inf
            if bbgt.size > 0:
                ixmin = np.maximum(bbgt[:, 0], bb[0])
                iymin = np.maximum(bbgt[:, 1], bb[1])
                ixmax = np.minimum(bbgt[:, 2], bb[2])
                iymax = np.minimum(bbgt[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                union = (bb[2] - bb[0]) * (bb[3] - bb[1]) + \
                        (bbgt[:, 2] - bbgt[:, 0] + 1.) * (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters

                ious = inters / union
                
                iou_max = np.max(ious)
                jmax = np.argmax(ious)

            if iou_max > iou_thresh:
                if not class_labels[index]['det'][jmax]:
                    TP[i] = 1
                    if prob > conf_thresh:
                        # print(iou_max, prob, bb[0:5])
                        tp += 1
                    class_labels[index]['det'][jmax] = True
                else:
                    FP[i] = 1
                    if prob > conf_thresh:
                        fp += 1
            else:
                FP[i] = 1
                if prob > conf_thresh:
                    fp += 1

        rec = tp / float(npos + np.finfo(np.float64).eps)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        rec_dict[cls] = [conf_thresh, rec]
        prec_dict[cls] = [conf_thresh, prec]

        TP = np.cumsum(TP)
        FP = np.cumsum(FP)
        REC = TP / float(npos + np.finfo(np.float64).eps)
        PREC = TP / np.maximum(TP + FP, np.finfo(np.float64).eps)
        AP = eval_ap(REC, PREC)

        print('{} & {:.2f}:'.format(class_map[cls], conf_thresh))
        print('\trec = {}/{} = {:.2%}'.format(tp, npos, rec))
        print('\tprec = {}/{} = {:.2%}'.format(tp, tp + fp, prec))
        print('\tap = {:.2%}'.format(AP))
    
        avg_rec.append(rec)
        avg_prec.append(prec)
        mAP.append(AP)

    avg_rec = np.mean(avg_rec)
    avg_prec = np.mean(avg_prec)
    mAP = np.mean(mAP)

    # print('average:')
    # print('\tavg_rec = {:.2%}'.format(avg_rec))
    # print('\tavg_prec = {:.2%}'.format(avg_prec))
    # print('\tmAP = {:.2%}'.format(mAP))

    return avg_rec, avg_prec, mAP, rec_dict, prec_dict

if __name__ == '__main__':
    # labels_dir = sys.argv[1]
    # file_pos = sys.argv[2]
    labels_dir = '/data0/TILT/720p/labels/'
    file_pos = 'output_720p_yolov3/pred.txt'

    curve_name = 'k.jpg'

    labels = load_labels(labels_dir)
    preds_pos = load_preds(file_pos)
   
    thresh = [x for x in np.arange(0, 1.0, 0.05)]
    
    y_rec = {}
    y_prec = {}
    y_chair = {}
    y_part = {}
    for t in thresh:
        print(t)
        avg_rec, avg_prec, mAP, rec_dict, prec_dict = eval(labels, preds_pos, 0.5, t)
        for cls, thresh_vs_rec in rec_dict.items():
            if cls not in y_rec:
                y_rec[cls] = []
            print(thresh_vs_rec[1])
            y_rec[cls].append(thresh_vs_rec[1])

        for cls, thresh_vs_prec in prec_dict.items():
            if cls not in y_prec:
                y_prec[cls] = []
            y_prec[cls].append(thresh_vs_prec[1])
    

   
    # for i, cls in enumerate(y_rec):
    #     plt.subplot(len(y_rec), 1, i + 1)
    for cls in [0]:

        plt.title(class_map[cls])
        plt.xlabel("T", fontsize=12)
        plt.ylabel("R/P/F", fontsize=12, rotation = 90)

        plt.plot(thresh, y_rec[cls], label='rec'+str(cls))
        plt.plot(thresh, y_prec[cls], label='prec'+str(cls))

        plt.legend()
        plt.grid(linestyle='--')
    plt.savefig(curve_name)

    # print('average:')
    # print('\tavg_rec = {:.2%}'.format(avg_rec))
    # print('\tavg_prec = {:.2%}'.format(avg_prec))
    # print('\tmAP = {:.2%}'.format(mAP))


