import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from src import model
from src import util
from src.body import Body
import json
import argparse


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--mode', default='validate', type=str)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


if args.mode=='annotate':
    cap = cv2.VideoCapture("rose_slow.mp4")
    body_estimation = Body('model/body_pose_model.pth')
    annotations = []
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, oriImg = cap.read()
        print(i)
        candidate, subset = body_estimation(oriImg)
        cv2.imwrite("images/rose_slow/%04d.jpg"%(i), oriImg)
        tempAnnotations = []
        for j in range(len(candidate)):
            tempAnnotations.append([candidate[j][0], candidate[j][1]])
        annotations.append({"path":"images/rose_slow/%04d.jpg"%(i), "keypoint":tempAnnotations})
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        with open('test.json', 'w') as f:
            json.dump(annotations, f, indent=4)
    cap.release()
    cv2.destroyAllWindows()

elif args.mode=='manual':
    json_open = open('data/rose_slow.json', 'r')
    json_load = json.load(json_open)
    print(len(json_load))
    annotations = []
    for i in range(100, len(json_load)):
        img = cv2.imread(json_load[i]['path'])
        if i%10==0:
            resizeImg = cv2.resize(img, (int(img.shape[:3][1]*3/4), int(img.shape[:3][0]*3/4)))
            box = cv2.selectROI('SiamMask', resizeImg, False, False)
            box = [int(box[0]*4/3), int(box[1]*4/3), int(box[2]*4/3), int(box[3]*4/3)]
            cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 255, 0))
            annotations.append({"path":json_load[i]['path'], "keypoint":json_load[i]['keypoint'], "tippoint":box})
            with open('rose_slow_enh.json', 'w') as f:
                json.dump(annotations, f, indent=4)

elif args.mode=='validate':
    json_open = open('data/rose_slow_enh.json', 'r')
    json_load = json.load(json_open)
    for i in range(len(json_load)):
        img = cv2.imread(json_load[i]['path'])
        for j in json_load[i]['keypoint']:
            cv2.circle(img, (int(j[0]), int(j[1])), 5, (255, 255, 255), thickness=-1)
        box = json_load[i]['tippoint']
        cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 255), 5)
        cv2.circle(img, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 10, (0, 0, 0), thickness=-1)
        cv2.imshow("img", img)
        cv2.waitKey(30)