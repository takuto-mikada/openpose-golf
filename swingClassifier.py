import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import json

from src import model
from src import util_enh as util
from src.body_enh import Body

cap = cv2.VideoCapture("images/rose/rose_01/rose_01.mp4")
body_estimation = Body('../data/model/body_pose_model.pth', "../data/golf_model/left_arm_model/checkpoint_iter_70.pth")

for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, oriImg = cap.read()
    featureImage, candidate, subset, ALL_PEAKS = body_estimation(oriImg)
    left_arm_keys = body_estimation.add_model(featureImage)
    

        
    
