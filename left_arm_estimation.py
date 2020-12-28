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
from src.left_arm_enh import Body


body_estimation = Body('../data/model/body_pose_model.pth', "../data/golf_model/left_arm_model/checkpoint_iter_70.pth")

for filename in glob.glob('../../Sample_trim/images/pose/*.jpg'):
    print(filename)
    videoName = filename.split("/")[-1].split(".")[0]

    oriImg = cv2.imread(filename)
    featureImage, candidate, subset, ALL_PEAKS = body_estimation(oriImg)
    print(ALL_PEAKS)
    cv2.imshow("A", oriImg)
    cv2.waitKey(0)
    # cv2.imwrite("../../Sample_trim/est_images/%s/%05d.jpg"%(videoName, i), canvas)
    # # left_arm_keys = body_estimation.add_model(featureImage)
    # rowData.update({i : KEYPOINT})
    # with open('../../Sample_trim/est_json/%s.json'%(videoName), 'w') as f:
    #     json.dump(rowData, f, indent=4)

        
    
