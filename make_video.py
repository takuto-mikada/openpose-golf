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


codec = cv2.VideoWriter_fourcc(*'mp4v')
video2 = cv2.VideoWriter("rose_slow_2.mp4", codec, 30, (1920, 1080))
video3 = cv2.VideoWriter("rose_slow_3.mp4", codec, 30, (1920, 1080))


for i in range(600):
    img = cv2.imread("images/rose_slow_est/%04d_2.jpg"%(i))
    img = cv2.resize(img, (1920, 1080))
    video2.write(img)
    img = cv2.imread("images/rose_slow_est/%04d_3.jpg"%(i))
    img = cv2.resize(img, (1920, 1080))
    video3.write(img)
    print(i)
    
video2.release()
video3.release()