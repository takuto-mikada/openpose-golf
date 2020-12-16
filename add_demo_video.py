import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from src import model
from src import util
from src.body_enh import Body
#from src.hand import Hand

import csv

cap = cv2.VideoCapture("rose_slow.mp4")
codec = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("rose_slow_estimation.mp4", codec, 30, (1920, 1080))
tip_data = open("rose_slow_tip_data.csv", "w")
writer = csv.writer(tip_data, lineterminator="\n")


body_estimation = Body('model/body_pose_model.pth', 'golf_model/checkpoint_iter_900.pth')

for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, oriImg = cap.read()
    print(i)
    with torch.no_grad():
        feature, candidate, subset = body_estimation(oriImg)
        tip = body_estimation.add_model(feature)
    
    cv2.imwrite("images/rose_slow_est/%04d_%d.jpg"%(i, 0), oriImg)
    tipImg = tip[10][0].to('cpu').detach().numpy().copy()[0]
    # print(tipImg.shape, tipImg.min(), tipImg.max())
    tipImg = np.resize(tipImg, (135, 240, 1))
    tipImg = (tipImg - tipImg.min())/(tipImg.max()-tipImg.min())*255
    tipImg = np.array(tipImg, dtype='uint8')
    cv2.imwrite("images/rose_slow_est/%04d_%d.jpg"%(i, 1), tipImg)
    
    tipImg = tip[10][0].to('cpu').detach().numpy().copy()[1]
    # print(tipImg.shape, tipImg.min(), tipImg.max())
    tipImg = np.resize(tipImg, (135, 240, 1))
    tipImg = (tipImg - tipImg.min())/(tipImg.max()-tipImg.min())*255
    tipImg = np.array(tipImg, dtype='uint8')
    cv2.imwrite("images/rose_slow_est/%04d_%d.jpg"%(i, 2), tipImg)
    
    tipImg = tip[9][0].to('cpu').detach().numpy().copy()
    # print(tipImg.shape, tipImg.min(), tipImg.max())
    tipImg = np.resize(tipImg, (135, 240, 1))
    tipImg = (tipImg - tipImg.min())/(tipImg.max()-tipImg.min())*255
    tipImg = np.array(tipImg, dtype='uint8')
    cv2.imwrite("images/rose_slow_est/%04d_%d.jpg"%(i, 3), tipImg)
    
    idx = np.unravel_index(np.argmax(tipImg), tipImg.shape)
    print(idx)
    tip_y = int(idx[0]*1080/135)
    tip_x = int(idx[1]*1920/240)
    writer.writerow([tip_x, tip_y])
    
    
    
    
    # cv2.imshow("A", canvas)
    # cv2.waitKey()
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    
    cv2.circle(canvas, (tip_x, tip_y), 10, (0, 255, 255), thickness=-1)
    # cv2.imwrite("images/rose_slow_est/%04d_%d.jpg"%(i, 0), canvas)
    video.write(canvas)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tip_data.close()
cap.release()
video.release()
cv2.destroyAllWindows()