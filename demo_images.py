import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import json

from src import model
from src import util_enh as util
from src.body_enh import Body
#from src.hand import Hand

image_path = ["hideki/hideki_01", "hideki/hideki_02", \
              "rose/rose_01", "rose/rose_02", \
              "tiger/tiger_01", "tiger/tiger_02", "tiger/tiger_03", \
              "mcllroy/mcllroy_01", "mcllroy/mcllroy_02", "mcllroy/mcllroy_04", \
              "spieth/spieth_02", "spieth/spieth_03", \
              "thomas/thomas_02"]

body_estimation = Body('model/body_pose_model.pth')

annotations = []
for i in image_path:
    for pose in ["address", "top", "finish"]:
        oriImg = cv2.imread("images/%s/%s.jpg"%(i, pose))
        print("images/%s/%s.jpg"%(i, pose))
        print(np.array(oriImg).shape)
        candidate, subset = body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas, connection = util.draw_bodypose(canvas, candidate, subset)
        print(connection)
        
        cv2.imwrite("images/classifier/%s_%s.jpg"%(i.split('/')[-1], pose), canvas)
        
        tempAnnotations = {}
        for j in range(len(connection)):
            tempAnnotations.update({"%s"%(connection[j][2]) : [connection[j][0], connection[j][1]]})
        annotations.append({"path":"images/%s/%s.jpg"%(i, pose), "keypoint":tempAnnotations})
        
        with open('classifire.json', 'w') as f:
            json.dump(annotations, f, indent=4)
        
    
