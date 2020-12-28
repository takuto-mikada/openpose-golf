import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import json
import glob
import os

from src import model
from src import util_enh as util
from src.body_enh import Body


NAME = ["IMG_4365", "IMG_4366", "IMG_4367", "IMG_4369", "IMG_4370", "IMG_4371", "IMG_4372", "IMG_4373", "IMG_4374"]
body_estimation = Body('../data/model/body_pose_model.pth')

for filename in glob.glob('../../Sample_trim/raw_video/*'):
    print(filename)
    videoName = filename.split("/")[-1].split(".")[0]
    if videoName in NAME:
        if not os.path.exists("../../Sample_trim/est_images/%s"%(videoName)):
            os.mkdir("../../Sample_trim/est_images/%s"%(videoName))
        cap = cv2.VideoCapture(filename)
        rowData = {}
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, oriImg = cap.read()
            featureImage, candidate, subset, ALL_PEAKS = body_estimation(oriImg)
            canvas = copy.deepcopy(oriImg)
            canvas, KEYPOINT = util.draw_bodypose(canvas, candidate, subset)
            cv2.imwrite("../../Sample_trim/est_images/%s/%05d.jpg"%(videoName, i), canvas)
            # left_arm_keys = body_estimation.add_model(featureImage)
            rowData.update({i : KEYPOINT})
            with open('../../Sample_trim/est_json/%s.json'%(videoName), 'w') as f:
                json.dump(rowData, f, indent=4)

        
    
