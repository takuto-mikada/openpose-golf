import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from src import model
from src import util
from src.body import Body

cap = cv2.VideoCapture("images/rose/rose_01/rose_01.mp4")

body_estimation = Body('model/body_pose_model.pth')

for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    

cap.release()
cv2.destroyAllWindows()