import copy
import json
import math
import os
import pickle

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset

BODY_PARTS_KPT_IDS = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
                      [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17], [4, 18], [7, 18]]
LEFT_ARM = [[0, 1], [1, 2], [2, 3], [3, 4]]


def get_mask(segmentations, mask):
    for segmentation in segmentations:
        rle = pycocotools.mask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        mask[pycocotools.mask.decode(rle) > 0.5] = 0
    return mask


class CocoTrainDataset(Dataset):
    def __init__(self, labels, images_folder, stride, sigma, paf_thickness, transform=None):
        super().__init__()
        self._images_folder = images_folder
        self._stride = stride
        self._sigma = sigma
        self._paf_thickness = paf_thickness
        self._transform = transform
        self._labels = json.load(open(labels, 'r'))

    def __getitem__(self, idx):
        label = copy.deepcopy(self._labels[idx])  # label modified in transform
        image = cv2.imread(os.path.join(self._images_folder, label['path']), cv2.IMREAD_COLOR)
        sample = {
            'label': label,
            'image': image,
            'path': os.path.join(self._images_folder, label['path']),
        }
        if self._transform:
            sample = self._transform(sample)

        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps

        paf_maps = self._generate_paf_maps(sample)
        sample['paf_maps'] = paf_maps

        # image = sample['image'].astype(np.float32)
        # image = (image - 128) / 256
        # sample['image'] = image.transpose((2, 0, 1))
        del sample['label']
        return sample

    def __len__(self):
        return len(self._labels)

    def _generate_keypoint_maps(self, sample):
        # n_keypoints = 19
        n_keypoints = 2
        n_rows, n_cols, _ = sample['image'].shape
        # keypoint_maps = np.zeros(shape=(n_keypoints + 1,
        #         n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg
        keypoint_maps = np.zeros(shape=(n_keypoints + 1, 135, 240), dtype=np.float32)
        
        keypoint = sample["label"]["tip"]
        self._add_gaussian(keypoint_maps[0], keypoint[0], keypoint[1], self._stride, self._sigma)
        # label = sample['label']
        # for keypoint_idx in range(18, n_keypoints):
        #     if keypoint_idx<18:
        #         keypoint = label['keypoint'][keypoint_idx]
        #     else:
        #         xc = int(label['tippoint'][0]+label['tippoint'][2]/2)
        #         yc = int(label['tippoint'][1]+label['tippoint'][3]/2)
        #         keypoint = [xc, yc]
        #     self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
        return keypoint_maps

    def _add_gaussian(self, keypoint_map, x, y, stride, sigma):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                    (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

    def _generate_paf_maps(self, sample):
        # n_pafs = len(BODY_PARTS_KPT_IDS)
        n_pafs = 1
        n_rows, n_cols, _ = sample['image'].shape
        paf_maps = np.zeros(shape=(n_pafs * 2, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)
        
        # print("sample keys : ", sample.keys())
        # print("sample label keys : ", sample["label"].keys())
        keypoint_a = sample["label"]["key"]["3"]
        xc = int(sample["label"]['tip'][0]+sample["label"]['tip'][2]/2)
        yc = int(sample["label"]['tip'][1]+sample["label"]['tip'][3]/2)
        keypoint_b = [xc, yc]
        self._set_paf(paf_maps,
                            keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                            self._stride, self._paf_thickness)
        # for paf_idx in range(20, n_pafs):
        #     if BODY_PARTS_KPT_IDS[paf_idx][0]<18:
        #         keypoint_a = [int(label['keypoint'][BODY_PARTS_KPT_IDS[paf_idx][0]][0]), int(label['keypoint'][BODY_PARTS_KPT_IDS[paf_idx][0]][1])]
        #     else:
        #         xc = int(label['tippoint'][0]+label['tippoint'][2]/2)
        #         yc = int(label['tippoint'][1]+label['tippoint'][3]/2)
        #         keypoint_a = [xc, yc]
        #     if BODY_PARTS_KPT_IDS[paf_idx][1]<18:
        #         keypoint_b = [int(label['keypoint'][BODY_PARTS_KPT_IDS[paf_idx][1]][0]), int(label['keypoint'][BODY_PARTS_KPT_IDS[paf_idx][1]][1])]
        #     else:
        #         xc = int(label['tippoint'][0]+label['tippoint'][2]/2)
        #         yc = int(label['tippoint'][1]+label['tippoint'][3]/2)
        #         keypoint_b = [xc, yc]
        #     self._set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
        #                     keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
        #                     self._stride, self._paf_thickness)
        return paf_maps

    def _set_paf(self, paf_map, x_a, y_a, x_b, y_b, stride, thickness):
        x_a /= stride
        y_a /= stride
        x_b /= stride
        y_b /= stride
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        _, h_map, w_map = paf_map.shape
        x_min = int(max(min(x_a, x_b) - thickness, 0))
        x_max = int(min(max(x_a, x_b) + thickness, w_map))
        y_min = int(max(min(y_a, y_b) - thickness, 0))
        y_max = int(min(max(y_a, y_b) + thickness, h_map))
        norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
        if norm_ba < 1e-7:  # Same points, no paf
            return
        x_ba /= norm_ba
        y_ba /= norm_ba

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)
                if d <= thickness:
                    paf_map[0, y, x] = x_ba
                    paf_map[1, y, x] = y_ba


class CocoValDataset(Dataset):
    def __init__(self, labels, images_folder):
        super().__init__()
        with open(labels, 'r') as f:
            self._labels = json.load(f)
        self._images_folder = images_folder

    def __getitem__(self, idx):
        file_name = self._labels['images'][idx]['file_name']
        img = cv2.imread(os.path.join(self._images_folder, file_name), cv2.IMREAD_COLOR)
        return {
            'img': img,
            'file_name': file_name
        }

    def __len__(self):
        return len(self._labels['images'])
