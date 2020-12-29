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


body_estimation = Body('../data/model/body_pose_model.pth', "../data/golf_model/left_arm_model/checkpoint_iter_70.pth", False)

for filename in glob.glob('../data/images/hideki/hideki_01/*.jpg'):
    print('../data/images/hideki/hideki_01/addres.jpg')
    videoName = filename.split("/")[-1].split(".")[0]

    oriImg = cv2.imread('../data/images/hideki/hideki_01/address.jpg')
    cv2.imshow("ori", cv2.resize(oriImg,(960,540)))
    featureImage, candidate, subset, ALL_PEAKS = body_estimation(oriImg, False)
    
    # cv2.imwrite("../../%05d.jpg"%(videoName, i), canvas)
    # # left_arm_keys = body_estimation.add_model(featureImage)
    # rowData.update({i : KEYPOINT})
    # with open('../..//%s.json'%(videoName), 'w') as f:
    #     json.dump(rowData, f, indent=4)

        
    
