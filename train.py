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
if torch.cuda.is_available():
    golf = DataParallel(golf).cuda()
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
            keypoint_maps = batch_data['keypoint_maps'][i]
            paf_maps = batch_data['paf_maps'][i]
            if torch.cuda.is_available():
                keypoint_maps = keypoint_maps.cuda(device)
                paf_maps = paf_maps.cuda(device)

            featureImage, candidate, subset, ALL_PEAKS = body(images)
            stages_output = golf(featureImage)

            

            # # keypoint map のアノテーション画像確認
            # for i in range(len(keypoint_maps)):
            #     KEY_img = keypoint_maps[i].to('cpu').detach().numpy().copy()
            #     cv2.imshow("S", KEY_img)
            #     cv2.waitKey()

            # # paf map のアノテーション画像確認
            # for i in range(len(paf_maps)):
            #     PAF_img = paf_maps[i].to('cpu').detach().numpy().copy()
            #     cv2.imshow("S", PAF_img)
            #     cv2.waitKey()

            for KEY in range(len(keypoint_maps)):
                for loss_idx in range(len(total_losses) // 2):
                    losses.append(l2_loss(stages_output[loss_idx * 2 + 1][0][KEY], keypoint_maps[KEY], images.shape[0]))
                    total_losses[loss_idx * 2 + 1] += losses[-1].item() / batches_per_iter
            
            for KEY in range(len(paf_maps)//2):
                for loss_idx in range(len(total_losses) // 2):
                    losses.append(l2_loss(stages_output[loss_idx * 2][0][KEY*2], paf_maps[KEY*2], images.shape[0]))
                    losses.append(l2_loss(stages_output[loss_idx * 2][0][KEY*2+1], paf_maps[KEY*2+1], images.shape[0]))
                    total_losses[loss_idx * 2] += losses[-2].item() / batches_per_iter
                    total_losses[loss_idx * 2] += losses[-1].item() / batches_per_iter
                    
            
        
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
        
        print(num_iter)

        if num_iter % log_after == 0:
            print('Iter: {}'.format(num_iter))
            for loss_idx in range(len(total_losses) // 2):
                print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
                    loss_idx + 1, total_losses[loss_idx * 2 + 1] / log_after,
                    loss_idx + 1, total_losses[loss_idx * 2] / log_after))
            for loss_idx in range(len(total_losses)):
                total_losses[loss_idx] = 0
        if num_iter % checkpoint_after == 0:
            snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
            print(snapshot_name)
            torch.save(golf.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict(),
                        # 'iter': num_iter,
                        # 'current_epoch': epochId},
                        snapshot_name)
        if num_iter % val_after == 0:
            print('Validation...')
            # candidate, subset, ALL_PEAKS = estimate(images, stages_output[11], stages_output[10])
            # canvas = copy.deepcopy(images)
            # canvas, KEYPOINT = util.draw_bodypose(canvas, candidate, subset)
            if not os.path.isfile("../data/golf_model/clab_tip_images/%05d"%(num_iter)):
                os.mkdir("../data/golf_model/clab_tip_images/%05d"%(num_iter))
            # cv2.imwrite("../data/golf_model/clab_tip_images/%05d/canvas.jpg"%(num_iter), canvas)
            
            keyImage = np.reshape(stages_output[11].to('cpu').detach().numpy().copy(), (1, len(keypoint_maps), 135, 240))
            keyImage = np.transpose(np.squeeze(keyImage), (1, 2, 0))*256
            pafImage = np.reshape(stages_output[10].to('cpu').detach().numpy().copy(), (1, len(paf_maps), 135, 240))
            pafImage = np.transpose(np.squeeze(pafImage), (1, 2, 0))*128+128


            for keys in range(len(keypoint_maps)):
                cv2.imwrite("../data/golf_model/clab_tip_images/%05d/key.jpg"%(num_iter), keyImage[:,:,keys])
            for pafs in range(len(paf_maps)//2):
                cv2.imwrite("../data/golf_model/clab_tip_images/%05d/map_x_%d.jpg"%(num_iter, pafs), pafImage[:,:,pafs*2])
                cv2.imwrite("../data/golf_model/clab_tip_images/%05d/map_y_%d.jpg"%(num_iter, pafs), pafImage[:,:,pafs*2+1])
            
            golf.train()



