from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from rcnn.center.centroidtracker import CentroidTracker
ct = CentroidTracker()
def distance(x1 , y1 , x2 , y2): 
    return math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) * 1.0) 
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img,bg):
    rects = []
    for detection in detections:
        if detection[0]==b'person':
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            ret=(xmin,ymin,xmax,ymax)
            rects.append(ret)

    objects = ct.update(rects)
    for e in range(len(objects)):
        # print(e)
        # exit()
        f=objects.copy()
        f=list(f.items())
        pt=[]
        for t in range(len(f)):
            np_to_list=np.array(f[t][1]).tolist()
            pt.append(np_to_list)
        m=pt.copy()
        a=m.pop(e)
        for x in m:

            distancea= distance(a[0],a[1],x[0],x[1])

            if distancea <45:
                text = "ID {}".format(pt.index(x))

                cv2.line(bg, (a[0], a[1]),(x[0],x[1]), (0, 0, 255), thickness=3, lineType=8)
                cv2.line(img, (a[0], a[1]),(x[0],x[1]), (255, 0, 0), thickness=3, lineType=8)

    for (objectID, centroid) in objects.items():


            cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.circle(bg, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture("/home/mustafa/TownCentreXVID.avi")

    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ff=cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(
        "output.avi", ff, 20.0,
        (832,416))
    out1 = cv2.VideoWriter(
        "output25.avi", ff, 20.0,
        (416,416))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.90)

        slice1 = np.zeros((int(416),int(416),3))
        bg = np.uint8(slice1)
        image = cvDrawBoxes(detections, frame_resized,bg)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(1/(time.time()-prev_time))
        numpy_vertical_concat = np.concatenate((image, bg), axis=1)
        out.write(numpy_vertical_concat)
        # out1.write(bg)
        cv2.imshow('Numpy Vertical', numpy_vertical_concat)

        cv2.imshow('Demo', image)
        cv2.imshow("draw",bg)
        cv2.waitKey(3)

    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()