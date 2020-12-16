import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import json
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from get_parameters import get_parameters_conv, get_parameters_conv_depthwise

from src import model
from src import util
from src.body_enh import Body
from src.model_enh import add_model
from coco import CocoTrainDataset
from loss import l2_loss


json_open = open('data/create/all_data.json', 'r')
data = json.load(json_open)

body = Body('model/body_pose_model.pth', "", False)
golf = add_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
batch_size = 5
num_workers = 1
stride = 8
sigma = 7
path_thickness = 1
batches_per_iter = 1
log_after = 10
checkpoint_after = 100
val_after = 100000
drop_after_epoch = [100, 200, 260]
checkpoints_folder = "golf_model/tip_model"
print(checkpoints_folder)

dataset = CocoTrainDataset('data/create/all_data.json', '',
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

optimizer = optim.Adam([
        {'params': get_parameters_conv(golf, 'weight')},
        {'params': get_parameters_conv(golf, 'bias')},
        {'params': get_parameters_conv_depthwise(golf, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(golf, 'bias'), 'weight_decay': 0},
    ], lr=4e-5, weight_decay=5e-4)
num_iter = 0
current_epoch = 0
drop_after_epoch = [100, 200, 260]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)
# if checkpoint_path:
#     checkpoint = torch.load(checkpoint_path)
#     if from_mobilenet:
#         load_from_mobilenet(net, checkpoint)
#     else:
#         load_state(net, checkpoint)
#         if not weights_only:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             scheduler.load_state_dict(checkpoint['scheduler'])
#             num_iter = checkpoint['iter']
#             current_epoch = checkpoint['current_epoch']

optimizer.step()
print(device)
#golf = DataParallel(golf).cuda()
golf = golf.cuda(device)
golf.train()
for epochId in range(current_epoch, 10000):
    scheduler.step()
    total_losses = [0, 0] * (6)
    batch_per_iter_idx = 0
    for batch_data in train_loader:
        if batch_per_iter_idx == 0:
            optimizer.zero_grad()

        losses = []
        for i in range(len(batch_data['path'])):
            images = cv2.imread(batch_data['path'][i])
            keypoint_maps = batch_data['keypoint_maps'][i].cuda(device)
            paf_maps = batch_data['paf_maps'][i].cuda(device)
            featureImage, candidate, subset = body(images, False)
            stages_output = golf(featureImage)
            
            # cv2.imwrite("images/rose_slow_ann/%d.jpg"%(0), images)
            # for k in [38, 39, 40, 41]:
            #     tipImg = paf_maps[k].to('cpu').detach().numpy().copy()
            #     print(tipImg.shape, tipImg.min(), tipImg.max())
            #     tipImg = np.resize(tipImg, (135, 240, 1))
            #     tipImg = (tipImg - tipImg.min())/(tipImg.max()-tipImg.min())*255
            #     tipImg = np.array(tipImg, dtype='uint8')
            #     cv2.imwrite("images/rose_slow_ann/%d.jpg"%(k), tipImg)
            # tipImg = keypoint_maps[18].to('cpu').detach().numpy().copy()
            # print(tipImg.shape, tipImg.min(), tipImg.max())
            # tipImg = np.resize(tipImg, (135, 240, 1))
            # tipImg = (tipImg - tipImg.min())/(tipImg.max()-tipImg.min())*255
            # tipImg = np.array(tipImg, dtype='uint8')
            # cv2.imwrite("images/rose_slow_ann/%d.jpg"%(50), tipImg)
            

       
            for loss_idx in range(len(total_losses) // 2):
                losses.append(l2_loss(stages_output[loss_idx * 2][0][0], paf_maps[0], images.shape[0]))
                losses.append(l2_loss(stages_output[loss_idx * 2][0][1], paf_maps[1], images.shape[0]))
                losses.append(l2_loss(stages_output[loss_idx * 2 + 1][0], keypoint_maps[0], images.shape[0]))
                total_losses[loss_idx * 2] += losses[-3].item() / batches_per_iter
                total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
                total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter
                
        loss = losses[0]
        for loss_idx in range(1, len(losses)):
            loss += losses[loss_idx]
        loss /= batches_per_iter
        loss.backward()
        batch_per_iter_idx += 1
        if batch_per_iter_idx == batches_per_iter:
            optimizer.step()
            batch_per_iter_idx = 0
            num_iter += 1
        else:
            continue

        if num_iter % log_after == 0:
            print('Iter: {}'.format(num_iter))
            for loss_idx in range(len(total_losses) // 2):
                print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
                    loss_idx + 1, total_losses[loss_idx * 2 + 1] / log_after,
                    loss_idx + 1, total_losses[loss_idx * 2] / log_after))
            for loss_idx in range(len(total_losses)):
                total_losses[loss_idx] = 0
            print(np.array(stages_output[10][0][0].to('cpu').detach().numpy().copy(), dtype=np.uint8).shape)
            cv2.imwrite("golf_model/tip_model_image/%04d_map1.jpg"%(num_iter), np.reshape(np.array(stages_output[10][0][0].to('cpu').detach().numpy().copy(), dtype=np.uint8), (135, 240, 1)))
            cv2.imwrite("golf_model/tip_model_image/%04d_map2.jpg"%(num_iter), np.reshape(np.array(stages_output[10][0][1].to('cpu').detach().numpy().copy(), dtype=np.uint8), (135, 240, 1)))
            cv2.imwrite("golf_model/tip_model_image/%04d_key.jpg"%(num_iter), np.reshape(np.array(stages_output[11][0].to('cpu').detach().numpy().copy(), dtype=np.uint8), (135, 240, 1)))
        if num_iter % checkpoint_after == 0:
            snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
            torch.save(golf.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict(),
                        # 'iter': num_iter,
                        # 'current_epoch': epochId},
                        snapshot_name)
        if num_iter % val_after == 0:
            print('Validation...')
            evaluate(val_labels, val_output_name, val_images_folder, golf)
            golf.train()



