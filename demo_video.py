import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from src import model
from src import util
from src.body import Body
#from src.hand import Hand

cap = cv2.VideoCapture("images/rose/rose_01/rose_01.mp4")

body_estimation = Body('model/body_pose_model.pth')
# hand_estimation = Hand('model/hand_pose_model.pth')

for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, oriImg = cap.read()
    print(i)
    candidate, subset = body_estimation(oriImg)
    # cv2.imshow("A", canvas)
    # cv2.waitKey()
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # canvas = cv2.resize(canvas, (960, 540))
    # cv2.imshow("S", canvas)
    # cv2.waitKey(0)
    input()

    # detect hand
    # hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    #for x, y, w, is_left in hands_list:
        # peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        #peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        #peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # all_hand_peaks.append(peaks)

    # canvas = util.draw_handpose(canvas, all_hand_peaks)

    print(np.array(canvas).shape)
    # video.write(canvas)
    cv2.imshow("a", canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()