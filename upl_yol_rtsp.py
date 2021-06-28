import argparse
import time
from pathlib import Path
from flask import Flask, render_template, Response
import os
import json
import cv2
import pytz 
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import pandas as pd
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

app = Flask(__name__)

# from pybo.models import Photos2

    



def detect():
    with open("passwd.json","r") as passwd:
        pwd = json.load(passwd)
    #parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    weights ="./yolov5s.pt"
    #source = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
    # source = "rtsp://smartinside:skku2400_@192.168.50.161:554/stream_ch00_1"  # file/folder, 0 for webcam
    source = pwd["source"]
    img_size = 640 #', type=int, default=640, help='inference size (pixels)')
    conf_thres = 0.4 #', type=float, default=0.25, help='object confidence threshold')
    iou_thres = 0.45 #', type=float, default=0.45, help='IOU threshold for NMS')
    device = ''
    view_img =  True #', action='store_true', help='display results')
    save_txt = True #', action='store_true', help='save results to *.txt')
    save_conf = True #', action='store_true', help='save confidences in --save-txt labels')
    nosave = True #', action='store_true', help='do not save images/videos')
    classes = [0, 1, 2, 3, 4] #', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    agnostic_nms = True # ', action='store_true', help='class-agnostic NMS')
    augment = True #', action='store_true', help='augmented inference')
    update = True #', action='store_true', help='update all models')
    project = 'runs/detect' #, help='save results to project/name')
    name = 'exp' #, help='save results to project/name')
    exist_ok = True #', action='store_true', help='existing project/name ok, do not increment')
    imgsz = img_size
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = True # source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #('rtsp://', 'rtmp://', 'http://'))

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    # Set Dataloader
    cudnn.benchmark = True 
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors=[[204,204,204], [0, 0, 255], [0, 153,204], [0, 153,204], [0, 204, 0]]
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    #0 = time.time() #start time
    #t_start = time.localtime()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        print(img)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        #t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        #t2 = time_synchronized()
        # Apply Classifier
#        if classify:
#            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            global im0
            #if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            #else:
            #    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s = s + f'{names[int(c)]}, {n} '  # add to string / 각 class의 이름과 n값 출                    
                    # #display object counting
                    # text = "Objects"
                    # text1 = ":{} ".format(s)
                    # im0 = cv2.putText(im0, text, (0, 70), cv2.FONT_HERSHEY_SIMPLEX,1,[255,144,30],3)
                    # im0 = cv2.putText(im0, text1, (-210, 110), cv2.FONT_HERSHEY_SIMPLEX,1,[255,144,30],3)

                    # smk_num,danger_num,nohelmet_num,nomask_num,safe_num=0,0,0,0,0
                    # for i in range(0,len(det)):
                    #     if(det[i][-1]==0): #smoke
                    #         smk_num+=1
                    #     if(det[i][-1]==1): #danger
                    #         danger_num+=1
                    #     if(det[i][-1]==2): #no-helmet
                    #         nohelmet_num+=1
                    #     if(det[i][-1]==3): #no-mask
                    #         nomask_num+=1
                    #     if(det[i][-1]==4): #safe
                    #         safe_num+=1
        
                    # if smk_num==0 and danger_num==0:
                    #     if nohelmet_num>=3 or nomask_num>=3 :
                    #         status = "danger"
                    #     if nohelmet_num==1 or nohelmet_num==2 : # Warning
                    #         status = "warning"
                    #     if nomask_num==1 or nomask_num==2 : # Warning
                    #         status = "warning"
                    #     if nohelmet_num==0 and nomask_num==0 and safe_num>=0: # Safe
                    #         status = "safe"
                    # elif smk_num > 0 or danger_num > 0 :
                    #         status = "danger"
                    # if status == "danger":
                    #     text2 = "Status: Danger !"
                    #     im0 = cv2.putText(im0, text2, (750, 50), cv2.FONT_HERSHEY_SIMPLEX,2,[0,0,255],4)
                    #     text2 = "Status: Warning !"
                    #     im0 = cv2.putText(im0, text2, (750, 50), cv2.FONT_HERSHEY_SIMPLEX,2,[0,255,255],4)
                    # if status == "safe":
                    #     text2 = "Status: Safe"
                    #     im0 = cv2.putText(im0, text2, (750, 50), cv2.FONT_HERSHEY_SIMPLEX,2,[0, 204, 0],4)                         

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                # Print time (inference + NMS)
                # print removed 
                    
                        _, jpeg = cv2.imencode('.jpg', im0)
                        frame = jpeg.tobytes()
                        #print(cv2.cuda.getCudaEnabledDeviceCount())
                        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/rtsp')
def video_feed():
    return Response(detect(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')
    
if __name__ == '__main__':

    # import sys
    # from os import path
    # print(path.dirname( path.dirname( path.abspath(__file__) ) ))
    # sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    # from models.experimental import attempt_load
    # from utils.datasets import LoadStreams, LoadImages
    # from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    # from utils.plots import plot_one_box
    # from utils.torch_utils import select_device, load_classifier, time_synchronized
    app.run(host='0.0.0.0', debug=True)
    
    