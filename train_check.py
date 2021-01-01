import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import json
import os
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.get_parameters import get_parameters_conv, get_parameters_conv_depthwise

from src import model
from src import util_enh as util
from src.body_enh import Body
from src.model_enh import add_model
from src.coco import CocoTrainDataset
from src.loss import l2_loss

from src.clab_tip import estimate


json_open = open('../data/json/top_finish.json', 'r')
data = json.load(json_open)

body = Body('../data/model/body_pose_model.pth')

golf = add_model()
# if torch.cuda.is_available():
#     golf = golf.cuda()
# golf_dict = util.add_transfer(golf, torch.load("../data/golf_model/clab_tip_model/checkpoint_iter_60.pth"))#,  map_location=torch.device('cpu')))
# golf.load_state_dict(golf_dict)
# golf.eval()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
batch_size = 1
num_workers = 1
stride = 8
sigma = 10
path_thickness = 1
batches_per_iter = 10
log_after = 1
checkpoint_after = 1
val_after = 1
drop_after_epoch = [100, 200, 260]
checkpoints_folder = "../data/golf_model/clab_tip_model"
print(checkpoints_folder)

dataset = CocoTrainDataset('../data/json/top_finish.json', '',
                               stride, sigma, path_thickness)
"""                               
                               transform=transforms.Compose([
                                   ConvertKeypoints(),
                                   Scale(),
                                   Rotate(pad=(128, 128, 128)),
                                   CropPad(pad=(128, 128, 128)),
                                   Flip()]))
"""                                   
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

num_iter = 0

if torch.cuda.is_available():
    golf = DataParallel(golf).cuda()
    golf = golf.cuda(device)

for batch_data in train_loader:
    for i in range(len(batch_data['path'])):
        images = cv2.imread(batch_data['path'][i])
        keypoint_maps = batch_data['keypoint_maps'][i]
        paf_maps = batch_data['paf_maps'][i]
        if torch.cuda.is_available():
            keypoint_maps = keypoint_maps.cuda(device)
            paf_maps = paf_maps.cuda(device)

        featureImage, candidate, subset, ALL_PEAKS = body(images)
        stages_output = golf(featureImage)
    
    print(num_iter)
    if num_iter % val_after == 0:
        print('Validation...')
        if not os.path.isdir("../data/golf_model/clab_tip_images/%05d"%(num_iter)):
            os.mkdir("../data/golf_model/clab_tip_images/%05d"%(num_iter))
        # cv2.imwrite("../data/golf_model/clab_tip_images/%05d/canvas.jpg"%(num_iter), canvas)
        
        for stg in range(6):
            keyImage = np.reshape(stages_output[stg*2+1].to('cpu').detach().numpy().copy(), (len(keypoint_maps), 135, 240))
            keyImage = np.transpose(keyImage, (1, 2, 0))
            if np.max(keyImage) - np.min(keyImage)!=0:
                keyImage = (keyImage - np.min(keyImage))*256/(np.max(keyImage) - np.min(keyImage))
            else:
                print("keyImage in stage%d not have value"%(stg))
            pafImage = np.reshape(stages_output[stg*2].to('cpu').detach().numpy().copy(), (len(paf_maps), 135, 240))
            pafImage = np.transpose(pafImage, (1, 2, 0))
            if np.max(pafImage) - np.min(pafImage)!=0:
                pafImage = (pafImage - np.min(pafImage))*256/(np.max(pafImage) - np.min(pafImage))
            else:
                print("pafImage in stage%d not have value"%(stg))

            for keys in range(len(keypoint_maps)):
                cv2.imwrite("../data/golf_model/clab_tip_images/%05d/key_stage%d.jpg"%(num_iter, stg), keyImage[:,:,keys])
            for pafs in range(len(paf_maps)//2):
                cv2.imwrite("../data/golf_model/clab_tip_images/%05d/map_x_%d_stage%d.jpg"%(num_iter, pafs, stg), pafImage[:,:,pafs*2])
                cv2.imwrite("../data/golf_model/clab_tip_images/%05d/map_y_%d_stage%d.jpg"%(num_iter, pafs, stg), pafImage[:,:,pafs*2+1])
        


