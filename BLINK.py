from __future__ import division
import cv2
import numpy as np
from matplotlib import pyplot as plt
from decimal import Decimal
import time


face_cascade = cv2.CascadeClassifier('C:\Users\Prajesh22\Desktop\BLINK\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\Users\Prajesh22\Desktop\BLINK\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

img_counter = 0
vid_counter = 0

while True:
    _, frame = cap.read()
    _, frame1 = cap1.read()
    frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x1,y1,w1,h1) in face:
        cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),10)
        roi_gray = gray[y1:y1+h1, x1:x1+w1]
        roi_color = frame[y1:y1+h1, x1:x1+w1]
    eye = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in eye:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        h2 = Decimal(h1)/Decimal(h)
        
    cv2.ellipse(frame,(315,220),(100,115),0,0,360,200,1)
    final_frame = cv2.hconcat((frame1,frame))
    cv2.imshow('Take Picture, Eye Detection',final_frame)
##    print("%2.3f" % h2, "%2.3f" % h1, "%2.3f" % h)
    print(x)
    print("-----------------")
    
    k = cv2.waitKey(27)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
##    elif 4.5<= h2 <= 5.5:
##        img_name = "BLINK_PICTURES_{}.png".format(img_counter)
##        cv2.imwrite(img_name, frame1)
##        print("{} written!".format(img_name))
##        img_counter += 1
    elif (x < 250):
        img_name = "BLINK_PIC_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame1)
        print("{} written!".format(img_name))
        img_counter += 1
##    elif (x > 350) & (4.5<= h2 <= 5.5):
##        out_video = cv2.VideoWriter('video.avi', -1, 25, (640, 480));
##        video.write(frame1)
##        out_vid = cv2.VideoWriter("BLINK_VID_{}.avi".format(vid_counter), -1, 20.0, (640,480))
##        out.write(out_vid, frame1)
##        print("{} written!".format(out_vid))
##        vid_counter += 1
##        if x < 325:
##            break

cap.release()
cv2.destroyAllWindows()
