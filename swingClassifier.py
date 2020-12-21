import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import json

from src import model
from src import util_enh as util
from src.body_enh import Body

cap = cv2.VideoCapture("../../Sample/IMG_4358.MOV")
body_estimation = Body('../data/model/body_pose_model.pth')

for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, oriImg = cap.read()
    featureImage, candidate, subset, ALL_PEAKS = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas, connection = util.draw_bodypose(canvas, candidate, subset)
    cv2.imshow("../data/pro/%05d.jpg"%(i), canvas)
    cv2.waitKey(0)
    # left_arm_keys = body_estimation.add_model(featureImage)
    print(ALL_PEAKS)

        
    
