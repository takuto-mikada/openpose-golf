import json
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


for filenames in glob.glob('../../Sample_trim/video/raw_video/*'):
    filename = filenames.split("/")[-1].split(".")[0]
    print(filename)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("../../Sample_trim/video/angle_video/%s.mp4"%(filename), codec, 30, (1920, 1080))
    DATA = json.load(open("../../Sample_trim/json/est_json/%s.json"%(filename), 'r'))
    cap = cv2.VideoCapture("../../Sample_trim/video/raw_video/%s.mp4"%(filename))
    if not os.path.exists("../../Sample_trim/images/pose/%s"%(filename)):
        os.mkdir("../../Sample_trim/images/pose/%s"%(filename))
    if not os.path.exists("../../Sample_trim/images/draw_angle/%s"%(filename)):
        os.mkdir("../../Sample_trim/images/draw_angle/%s"%(filename))

    time = []
    angleRow = []
    changePhase = []
    phase = 0
    changePhase.append(0)
    center = (0, 0)
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, img = cap.read()

        oriImg = img
        if "1" in DATA["%s"%(i)].keys() and "8" in DATA["%s"%(i)].keys() and "11" in DATA["%s"%(i)].keys():
            cx1 = int((DATA["%s"%(i)]["8"][0] + DATA["%s"%(i)]["11"][0])/2)
            cy1 = int((DATA["%s"%(i)]["8"][1] + DATA["%s"%(i)]["11"][1])/2)
            cx2 = int((DATA["%s"%(i)]["1"][0] + cx1)/2)
            cy2 = int((DATA["%s"%(i)]["1"][1] + cy1)/2)
            center = (cx2, cy2)
        
        if center[0]!=0 and center[0]!=0:
            cv2.circle(img, (cx2, cy2), 10, (0, 255, 0), thickness=-1)

            if "4" in DATA["%s"%(i)].keys():
                rist = (int(DATA["%s"%(i)]["4"][0]), int(DATA["%s"%(i)]["4"][1]))
                cv2.circle(img, rist, 10, (0, 0, 255), thickness=-1)
            if "7" in DATA["%s"%(i)].keys():
                rist = (int(DATA["%s"%(i)]["7"][0]), int(DATA["%s"%(i)]["7"][1]))
                cv2.circle(img, rist, 10, (255, 0, 0), thickness=-1)
            if "4" in DATA["%s"%(i)].keys() and "7" in DATA["%s"%(i)].keys():
                rist = (int((DATA["%s"%(i)]["4"][0] + DATA["%s"%(i)]["7"][0])/2), int((DATA["%s"%(i)]["4"][1] + DATA["%s"%(i)]["7"][1])/2))
            cv2.circle(img, rist, 10, (0, 255, 0), thickness=-1)

            cv2.line(img, rist, (cx2, cy2), (0, 255, 0), thickness=5, lineType=cv2.LINE_4)
            video.write(img)

            angle = (rist[0] - cx2, -(rist[1] - cy2))
            angle = math.atan2(angle[1], angle[0]) * 180 / math.pi
            time.append(i)
            angleRow.append(angle)
            with open("../../Sample_trim/json/angle/%s.json"%(filename), 'w') as f:
                json.dump(angleRow, f, indent=4)

            cv2.imwrite("../../Sample_trim/images/draw_angle/%s/%04d.jpg"%(filename, i), img)
            # img = cv2.resize(img, (1280, 640))
            # cv2.imshow("A", img)
            # cv2.waitKey(30)

            if phase==0 and angle<-150:
                phase = 1
                changePhase.append(i)
            elif phase==1 and 0<=angle and angle<150:
                phase = 2
                changePhase.append(i)
            elif phase==2 and 150<angle:
                phase = 3
                changePhase.append(i)
            elif phase==3 and -150<angle and angle<=0:
                phase = 4
                changePhase.append(i)
    video.release()

        
    changePhase.append(i)

    print(changePhase)

    plt.plot(time, angleRow)
    plt.savefig("../../Sample_trim/images/pose/%s/angle.png"%(filename))
    # plt.show()
    plt.gca().clear()
    
    if len(changePhase)==6:
        # top pose detection
        topPhase = {}
        for i in range(changePhase[2], changePhase[3]):
            topPhase.update({i : angleRow[i]})
        score = sorted(topPhase.items(), key=lambda x:x[1])
        print("top : ", score[0][0])

        diff = []
        preAngle = 0
        for i in range(len(angleRow)):
            angle = angleRow[i]
            diff.append(angle - preAngle)
            preAngle = angle

        # plt.plot(diff)
        # plt.ylim(-20, 20)
        # plt.show()
        # plt.gca().clear()
        # plt.savefig("diff.png")
        

        addressPhase = []
        for i in range(changePhase[0], changePhase[1]):
            if np.absolute(diff[i])<10:
                addressPhase.append(i)
        print("address : ", addressPhase[int(len(addressPhase)/2)])

        finishPhase = []
        for i in range(changePhase[4], changePhase[-1]):
            if np.absolute(diff[i])<10 and angleRow[i]>90:
                finishPhase.append(i)
        print("finish : ", finishPhase[int(len(finishPhase)/2)])


        cap = cv2.VideoCapture("../../Sample_trim/video/raw_video/%s.mp4"%(filename))
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, img = cap.read()
            if i==score[0][0]:
                cv2.imwrite("../../Sample_trim/images/pose/%s_top.jpg"%(filename), img)
            # elif i==addressPhase[int(len(addressPhase)/2)]:
            elif i==addressPhase[10]:
                cv2.imwrite("../../Sample_trim/images/pose/%s_address.jpg"%(filename), img)
            elif i==finishPhase[-10]:
                cv2.imwrite("../../Sample_trim/images/pose/%s_finish.jpg"%(filename), img)