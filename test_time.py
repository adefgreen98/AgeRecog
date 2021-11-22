
# The video-capture in this script follows the same structure as in 'realtime_demo.py': see there for better understanding

import os
import time
import numpy as np

import matplotlib.pyplot as plt
import cv2

from realtime_demo import prepare_model, predict

# Also used to get the number of parameters
from net import get_n_params

# REQUIRED PATHS
std_mnet_path = ''
resnet_path = ''
mnet_path = ''
video_path = ''

std_mnet = prepare_model('mobilenet', std_mnet_path, 117).cpu()
resnet = prepare_model('resnet18', resnet_path, 117).cpu()
mnet = prepare_model('2classifiermn', mnet_path, 117).cpu()

capturer = cv2.VideoCapture(video_path)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

std_mnet_times = []
resnet_times = []
mnet_times = []



fw = int(capturer.get(3))
fh = int(capturer.get(4))
fps = round(capturer.get(cv2.CAP_PROP_FPS))

lab = '0'
hop = round(fps * 0.25)
curr_frame = 0

while True:
    
    ret, frame = capturer.read()
    if not ret:
        break

    if curr_frame % hop == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1)
        toclassify = None            
        for x, y, w, h in faces:
            x = x-10
            y = y-10
            w = w + 10*2
            h = h + 10*2

            if w / fw > 1/12:
                toclassify = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                toclassify = toclassify[y:y+h, x:x+w]
                
                # try/catch required when face rectangle exceeds from the image (problems in transforming from cv2 to PIL)
                try:
                    lab, dt = predict(toclassify, std_mnet, True)
                    print('std_mnet: ', dt)
                    std_mnet_times.append(round(dt, 2))

                    lab, dt = predict(toclassify, resnet, True)
                    print('resnet: ', dt)
                    resnet_times.append(round(dt, 2))
                
                    lab,dt = predict(toclassify, mnet, True)    
                    print('mnet: ', dt)
                    mnet_times.append(round(dt, 2))
                except ValueError:
                    pass
    curr_frame += 1


# excluding first time because were taken uncorrectly
std_mnet_times = std_mnet_times[3:]
mnet_times = mnet_times[3:]
resnet_times = resnet_times[3:]

capturer.release()
cv2.destroyAllWindows()

fig = plt.figure()
graph = fig.add_subplot(111)
graph.plot(list(range(len(std_mnet_times))), std_mnet_times, label='Mobilenet(standard)')
graph.plot(list(range(len(resnet_times))), resnet_times, label='Resnet')
graph.plot(list(range(len(mnet_times))), mnet_times, label='MobileNet(custom)')
graph.set_ylabel('Tempo di inferenza (ms)')
graph.set_xlabel('Iterazione')
graph.legend()

fig.savefig('times.png')

print('std_mnet params:', get_n_params(std_mnet))
print('mnet params:', get_n_params(mnet))
print('resnet params: ', get_n_params(resnet))