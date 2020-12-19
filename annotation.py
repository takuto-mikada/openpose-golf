import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import math

from src import model
from src import util
from src.body import Body
import json
import argparse
import os
import glob



colors = [[255, 0, 0], [255, 85, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [255, 170, 0], [255, 255, 0], [170, 255, 0], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--mode', default='validate', type=str)
args = parser.parse_args()
print(args.mode)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


if args.mode=='annotate':
    cap = cv2.VideoCapture("rose_slow.mp4")
    body_estimation = Body('model/body_pose_model.pth')
    annotations = []
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, oriImg = cap.read()
        print(i)
        candidate, subset = body_estimation(oriImg)
        cv2.imwrite("images/rose_slow/%04d.jpg"%(i), oriImg)
        tempAnnotations = []
        for j in range(len(candidate)):
            tempAnnotations.append([candidate[j][0], candidate[j][1]])
        annotations.append({"path":"images/rose_slow/%04d.jpg"%(i), "keypoint":tempAnnotations})
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        with open('test.json', 'w') as f:
            json.dump(annotations, f, indent=4)
    cap.release()
    cv2.destroyAllWindows()

elif args.mode=='tip_create':
    player = "mcllroy"
    number = 2
    videoPath = "images/%s/%s_%02d/%s_%02d.mp4"%(player, player, number, player, number)
    dataPath = "data/create/%s_%02d.json"%(player, number)
    cap = cv2.VideoCapture(videoPath)
    saveData = []
    start = 0
    if os.path.isfile(dataPath):
        json_open = open(dataPath, 'r')
        json_load = json.load(json_open)
        start = int(json_load[-1]["path"].split("/")[-1].split(".")[0]) + 1
        for row in range(len(json_load)):
            saveData.append(json_load[row])

    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, oriImg = cap.read()
        if i%3==0 and i>=int(start):
            print(i, start)
            tempData = {"path" : "images/create/%s/%s_%02d/%04d.jpg"%(player, player, number, i)}
            cv2.imwrite("images/create/%s/%s_%02d/%04d.jpg"%(player, player, number, i), oriImg)
            resizeImg = cv2.resize(oriImg, (int(oriImg.shape[:3][1]*3/4), int(oriImg.shape[:3][0]*3/4)))
            box = cv2.selectROI('SiamMask', resizeImg, False, False)
            box = [int(box[0]*4/3), int(box[1]*4/3), int(box[2]*4/3), int(box[3]*4/3)]
            tempData.update({"tip" : box})
            saveData.append(tempData)
            with open(dataPath, 'w') as f:
                json.dump(saveData, f, indent=4)

elif args.mode=='key_create':
    player = "spieth"
    number = 3
    videoPath = "images/%s/%s_%02d/%s_%02d.mp4"%(player, player, number, player, number)
    dataPath = "data/create/%s_%02d.json"%(player, number)
    json_open = open(dataPath, 'r')
    saveData = json.load(json_open)

    for i in range(len(saveData)):
        if "key" not in saveData[i].keys():
            oriImg = cv2.imread(saveData[i]["path"])
            resizeImg = cv2.resize(oriImg, (int(oriImg.shape[:3][1]*3/4), int(oriImg.shape[:3][0]*3/4)))
            box = cv2.selectROI('SiamMask', resizeImg, False, False)
            box = [int(box[0]*4/3), int(box[1]*4/3), int(box[2]*4/3), int(box[3]*4/3)]
            cv2.circle(oriImg, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 10, (0, 255, 0), thickness=-1)
            saveData[i].update({"key" : { "%d"%(3) : [int(box[0]+box[2]/2), int(box[1]+box[3]/2)]}})
            # for j in [1, 2, 3]:
            #     resizeImg = cv2.resize(oriImg, (int(oriImg.shape[:3][1]*3/4), int(oriImg.shape[:3][0]*3/4)))
            #     box = cv2.selectROI('SiamMask', resizeImg, False, False)
            #     box = [int(box[0]*4/3), int(box[1]*4/3), int(box[2]*4/3), int(box[3]*4/3)]
            #     saveData[i]["key"].update({ "%d"%(j) : [int(box[0]+box[2]/2), int(box[1]+box[3]/2)]})
            #     cv2.circle(oriImg, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 10, (0, 255, 0), thickness=-1)
        with open(dataPath, 'w') as f:
            json.dump(saveData, f, indent=4)

elif args.mode=='test':
    dataPath = "../data/json/top_finish.json"
    saveData = []
    pathList = []
    if os.path.exists(dataPath):
        tempData = json.load(open(dataPath, 'r'))
        for i in range(len(tempData)):
            saveData.append(tempData[i])
            pathList.append(tempData[i]["path"])
    i = 0
    print(len(pathList))
    for path in glob.glob("../data/images/left_annotation/*"):
        tempData = {}
        print(i)
        if path not in pathList:
            oriImg = cv2.imread(path)
            resizeImg = cv2.resize(oriImg, (int(oriImg.shape[:3][1]*3/4), int(oriImg.shape[:3][0]*3/4)))
            box = cv2.selectROI('SiamMask', resizeImg, False, False)
            box = [int(box[0]*4/3), int(box[1]*4/3), int(box[2]*4/3), int(box[3]*4/3)]
            cv2.circle(oriImg, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 10, (0, 255, 0), thickness=-1)
            tempData.update({"path" : path, 
                             "tip" : [int(box[0]+box[2]/2), int(box[1]+box[3]/2)]})
            tempKey = []
            for j in [0, 1, 2, 3]:
                resizeImg = cv2.resize(oriImg, (int(oriImg.shape[:3][1]*3/4), int(oriImg.shape[:3][0]*3/4)))
                box = cv2.selectROI('SiamMask', resizeImg, False, False)
                box = [int(box[0]*4/3), int(box[1]*4/3), int(box[2]*4/3), int(box[3]*4/3)]
                tempKey.append({"%d"%(j) : [int(box[0]+box[2]/2), int(box[1]+box[3]/2)]})
                cv2.circle(oriImg, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 10, (0, 255, 0), thickness=-1)
            tempData.update({"key" : tempKey})
            saveData.append(tempData)
        with open(dataPath, 'w') as f:
            json.dump(saveData, f, indent=4)
        i+=1  

elif args.mode=='validate':
    json_open = open('data/classifire_tip.json', 'r')
    json_load = json.load(json_open)
    for i in range(len(json_load)):
        img = cv2.imread(json_load[i]['path'])
        print(json_load[i]['keypoint'], len(json_load[i]['keypoint']))
        for j in json_load[i]['keypoint']:
            cv2.circle(img, (int(j[0]), int(j[1])), 10, (0, 0, 255), thickness=-1)
        box = json_load[i]['tippoint']
        cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 5)
        cv2.circle(img, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 10, (0, 0, 0), thickness=-1)
        img = cv2.resize(img, (960, 540))
        cv2.imshow("img", img)
        cv2.waitKey(0)

elif args.mode=='add':
    json_open = open('data/classifire_add.json', 'r')
    json_load = json.load(json_open)
    for i in range(len(json_load)):
        print(json_load[i]['keypoint'].keys())
        while(1):
            img = cv2.imread(json_load[i]['path'])
            for j in json_load[i]['keypoint'].keys():
                cv2.circle(img, (int(json_load[i]['keypoint'][j][0]), int(json_load[i]['keypoint'][j][1])), 10, colors[int(j)], thickness=-1)
            if "tip" in json_load[i].keys():
                box = json_load[i]['tip']
                # cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 5)
                cv2.circle(img, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 10, (255, 0, 0), thickness=-1)

            resizeImg = cv2.resize(img, (int(img.shape[:3][1]*3/4), int(img.shape[:3][0]*3/4)))
            box = cv2.selectROI('SiamMask', resizeImg, False, False)
            box = [int(box[0]*4/3), int(box[1]*4/3), int(box[2]*4/3), int(box[3]*4/3)]
            num = input("what's number of joint : ")
            if num=="exit":
                break
            elif num.isdigit():
                if int(num)<17:
                    json_load[i]['keypoint'].update({num : [int(box[0]+box[2]/2), int(box[1]+box[3]/2)]})
                else:
                    print("not number in keys")
            elif num=="tip":
                json_load[i].update({"tip" : box})
        with open('data/classifire_add_v2.json', 'w') as f:
            json.dump(json_load, f, indent=4)
            

elif args.mode=='calculate':
    json_open = open('data/classifire_add_v2.json', 'r')
    json_load = json.load(json_open)
    List5 = [[8,9], [9,10], [11,12], [12,13], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7]]
    List6 = [[8,9], [9,10], [11,12], [12,13], [1,5], [5,6], [6,7]]
    List7 = [["center", "8-9"], ["center", "11-12"], ["8-9", "9-10"], ["11-12", "12-13"], \
             ["center", "1-2"], ["center", "1-5"], ["1-2", "2-3"], ["2-3", "3-4"], ["1-5", "5-6"], ["5-6", "6-7"], ["6-7", "7-tip"]]
    List8 = [["center", "8-9"], ["center", "11-12"], ["8-9", "9-10"], ["11-12", "12-13"], \
             ["center", "1-5"], ["1-5", "5-6"], ["5-6", "6-7"], ["6-7", "7-tip"]]
    List7 = [["center", "8-9"], ["center", "11-12"], ["center", "9-10"], ["center", "12-13"], \
             ["center", "1-2"], ["center", "1-5"], ["center", "2-3"], ["center", "3-4"], ["center", "5-6"], ["center", "6-7"], ["center", "7-tip"]]
    List8 = [["center", "8-9"], ["center", "11-12"], ["center", "9-10"], ["center", "12-13"], \
             ["center", "1-5"], ["center", "5-6"], ["center", "6-7"], ["center", "7-tip"]]
    for i in range(len(json_load)):
        img = cv2.imread(json_load[i]['path'])
        vector = []
        angle = []
        keys = json_load[i]['keypoint']
        center = [(keys["8"][0] + keys["11"][0]) / 2 ,(keys["8"][1] + keys["11"][1]) / 2 ]
        cv2.circle(img, (int(center[0]), int(center[1])), 10, (255, 0, 0), thickness=-1)
        cv2.line(img, (int(center[0]), int(center[1])), (int(keys["1"][0]), int(keys["1"][1])), (0, 0, 255), thickness=5, lineType=cv2.LINE_4)
        baseVector = [ int(center[0]) - int(keys["1"][0]), int(center[1]) - int(keys["1"][1]) ]
        baseLength = math.sqrt(baseVector[0]*baseVector[0] + baseVector[1]*baseVector[1])

        if json_load[i]['path'].split('/')[-1]=="address.jpg":
            LIST = List5
        else:
            LIST = List6
        vectors = {}
        vectors.update({"center" : [baseVector[0], baseVector[1]]})
        for point in LIST:
            cv2.line(img, (int(keys["%s"%(point[0])][0]), int(keys["%s"%(point[0])][1])), (int(keys["%s"%(point[1])][0]), int(keys["%s"%(point[1])][1])), (0, 0, 255), thickness=5, lineType=cv2.LINE_4)
            vector = [ int(keys["%s"%(point[1])][0]) - int(keys["%s"%(point[0])][0]), int(keys["%s"%(point[1])][1]) - int(keys["%s"%(point[0])][1]) ]
            vectors.update({"%d-%d"%(point[0], point[1]) : [vector[0], vector[1]]})
            theta = math.acos((vector[0]*baseVector[0] + vector[1]*baseVector[1]) / (math.sqrt(vector[0]*vector[0] + vector[1]*vector[1]) * math.sqrt(baseVector[0]*baseVector[0] + baseVector[1]*baseVector[1])))
            # print(theta*180/math.pi)
            cv2.circle(img, (int(keys["%s"%(point[0])][0]), int(keys["%s"%(point[0])][1])), 10, (255, 0, 0), thickness=-1)
            cv2.circle(img, (int(keys["%s"%(point[1])][0]), int(keys["%s"%(point[1])][1])), 10, (255, 0, 0), thickness=-1)
        
        box = json_load[i]['tip']
        tippoint = [int(box[0]+box[2]/2), int(box[1]+box[3]/2)]
        cv2.circle(img, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 10, (0, 255, 0), thickness=-1)
        cv2.line(img, (int(keys["7"][0]), int(keys["7"][1])), (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), (0, 0, 255), thickness=5, lineType=cv2.LINE_4)
        vector = [int(box[0]+box[2]/2)-json_load[i]['keypoint']['7'][0], int(box[1]+box[3]/2)-json_load[i]['keypoint']['7'][1]]
        vectors.update({"7-tip" : [vector[0], vector[1]]})
        json_load[i].update({"vector" : vectors})

        angle = {}
        if json_load[i]['path'].split('/')[-1]=="address.jpg":
            LIST = List7
        else:
            LIST = List8
        for j in LIST:
            vector = json_load[i]["vector"]
            theta = math.acos((vector[j[0]][0]*vector[j[1]][0] + vector[j[0]][1]*vector[j[1]][1]) / \
                    (math.sqrt(vector[j[0]][0]*vector[j[0]][0] + vector[j[0]][1]*vector[j[0]][1]) * math.sqrt(vector[j[1]][0]*vector[j[1]][0] + vector[j[1]][1]*vector[j[1]][1]))) * 180/math.pi
            angle.update({"%s,%s"%(j[0], j[1]) : theta})
        
        above_vector = [tippoint[0] - json_load[i]["keypoint"]["1"][0], tippoint[1] - json_load[i]["keypoint"]["1"][1]]
        above = math.sqrt(above_vector[0]*above_vector[0] + above_vector[1]*above_vector[1]) / baseLength * 50
        under_vector = [tippoint[0] - center[0], tippoint[1] - center[1]]
        under = math.sqrt(under_vector[0]*under_vector[0] + under_vector[1]*under_vector[1]) / baseLength * 50
        angle.update({"above_lingth" : above})
        angle.update({"under_lingth" : under})

        json_load[i].update({"angle" : angle})
        
        with open('data/classifire_angle.json', 'w') as f:
            json.dump(json_load, f, indent=4)
        
        img = cv2.resize(img, (960, 540))
        cv2.imwrite("images/estimate/%s_%s.jpg"%(json_load[i]["path"].split('/')[-2], json_load[i]["path"].split('/')[-1]), img)

elif args.mode=='classify':
    json_open = open('data/classifire_angle.json', 'r')
    json_load = json.load(json_open)

    player = {}
    for i in range(0, len(json_load), 3):
        player.update({json_load[i]["path"].split('/')[-2] : {"address" : json_load[i],
                                                              "top" : json_load[i+1],
                                                              "finish" : json_load[i+2]}})
    
    # print(player.keys())
    # target_name = input("input player name : ")
    for target_name in player.keys():
        target_data = player[target_name]
        save_diff = {}
        for i in player.keys():
            if i != target_name:
                sum_diff = 0
                for j in player[i].keys():
                    for row in player[i][j]["angle"].keys():
                        diff = target_data[j]["angle"][row] - player[i][j]["angle"][row]
                        sum_diff += math.sqrt(diff*diff)
                save_diff.update({i : sum_diff})
        score = sorted(save_diff.items(), key=lambda x:x[1])
        print(target_name)
        print(score)
        
