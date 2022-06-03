# -*- coding: utf-8 -*-
"""
Created on Fri May 13 08:23:50 2022

@author: Jean-Marie Hembach und Mariele Donoso Olave
"""

import cv2
vidcap = cv2.VideoCapture('./data_samples/05.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("./data_samples/Bilder/image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 1/29.60 #//it will capture image in each 0.5 second
count=1848
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)