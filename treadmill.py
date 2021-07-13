from __future__ import print_function
import argparse
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import subprocess
import imutils
from imutils.video import FPS
from imutils.video.pivideostream import PiVideoStream
import RPi.GPIO as GPIO
from gpiozero import Button

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO_calibrate = 16
GPIO.setup(GPIO_calibrate, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def ticcmd(*args):
    return subprocess.check_output(['ticcmd'] + list(args))

ticcmd('--reset')
ticcmd('--step-mode', str(int(4)))
ticcmd('--max-accel', str(int(1500000)))
ticcmd('--max-decel', str(int(1500000)))
ticcmd('--max-speed', str(int(18000000)))
ticcmd('--current', str(int(320)))
tic_base_speed = 5000000
tic_diff = 20000

camera=PiCamera()
camera.resolution = (320,240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
stream = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
camera.close()

vs=PiVideoStream().start()
time.sleep(2.0)
runs = 0
while True:

    x_value = 0

    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if runs == 0 or GPIO.input(GPIO_calibrate)==False:
        ret1, mask1 = cv2.threshold(grayscaled, 80, 255, cv2.THRESH_BINARY_INV)
        runs+=1
    cut_frame = cv2.bitwise_or(grayscaled, mask1)
    ret2, mask2 = cv2.threshold(cut_frame, 50,255, cv2.THRESH_BINARY)
    mask2 = cv2.bitwise_not(mask2)

    cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    minimum_area = 300
    for c in cnts:
        area = cv2.contourArea(c)
        if area > minimum_area:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 10, (36,255,12), 2)
            x,y,w,h = cv2.boundingRect(c)
            x_value = x
            cv2.putText(frame, 'X value: {}'.format(x), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            cv2.putText(frame, 'Y value: {}'.format(y), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            break

    if 0 < x_value < 300:
        speed = tic_base_speed +((300-x_value) * tic_diff)

    elif 300 <= x_value < 550:
        speed = tic_base_speed
    else:
        speed = 0
        x_value = 0

    ticcmd('--energize')
    #Slows it down
    ticcmd('--exit-safe-start', '--velocity', str(speed))
    cv2.putText(frame, 'Belt Speed: {} mm/s'.format(speed/42500), (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (96, 55, 212), 2)

    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Frame", frame)
    cv2.imshow("Threshold1", mask1)
    cv2.imshow("Threshold2", mask2)
    cv2.imshow("Cut", cut_frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cv2.destroyAllWindows()
GPIO.cleanup()
ticcmd('--enter-safe-start')
ticcmd('--deenergize')
vs.stop()