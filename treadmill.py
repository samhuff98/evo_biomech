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
from gpiozero import MCP3008
import csv

#setting up GPIO pins
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO_calibrate = 16
GPIO_LED_blue = 18
GPIO_manual_mode = 22
GPIO.setup(GPIO_calibrate, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(GPIO_LED_blue, GPIO.OUT)
GPIO.setup(GPIO_manual_mode, GPIO.IN, pull_up_down=GPIO.PUD_UP)
pot1 = MCP3008(0)
pot2 = MCP3008(1)

#setting up stepper motor
def ticcmd(*args):
    return subprocess.check_output(['ticcmd'] + list(args))
ticcmd('--reset')
ticcmd('--step-mode', str(int(4)))
ticcmd('--max-accel', str(int(1500000)))
ticcmd('--max-decel', str(int(1500000)))
ticcmd('--max-speed', str(int(18000000)))
ticcmd('--current', str(int(320)))
ticcmd('--energize')
tic_base_speed = 3000000
tic_max_speed = 20400000
tic_diff = 20000
speed=0

#setting up raspberry pi camera module
camera=PiCamera()
camera.resolution = (320,240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320,240))
stream = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
camera.close()
vs=PiVideoStream().start()
time.sleep(2.0)
runs = 0

#setup to save video footage and insect positions
filename = '/home/pi/Documents/evo_biomech/data.csv'
with open(filename, 'ab',) as csvfile:
    row_1 = ['X_values', 'Y_Values']
    csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(row_1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/pi/Documents/evo_biomech/video.avi', fourcc, 20.0, (1440, 900))

while True:
    x_value = 0
    #updating base speed based on potentiometer
    tic_base_speed = round(3000000+(4000000*(pot1.value)))
    frame = vs.read()
    #creating blank array for interface
    ndarray = np.full((900,1440,3), 20, dtype=np.uint8)
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5,5), np.uint8)
    #masking out the background
    if runs < 20 or GPIO.input(GPIO_calibrate)==False:
        GPIO.output(GPIO_LED_blue, True)
        ret1, mask1 = cv2.threshold(grayscaled, 120, 255, cv2.THRESH_BINARY_INV)
        mask1 = cv2.dilate(mask1, kernel, iterations=1)
        runs+=1
    else:
        GPIO.output(GPIO_LED_blue, False)

    #insect detection mask
    cut_frame = cv2.bitwise_or(grayscaled, mask1)
    ret2, mask2 = cv2.threshold(cut_frame, 100 ,255, cv2.THRESH_BINARY)
    mask2 = cv2.bitwise_not(mask2)

    #mask2 = cv2.dilate(mask2, kernel, iterations=1)
    mask1_disp = cv2.merge([mask1,mask1,mask1])
    mask2_disp = cv2.merge([mask2,mask2,mask2])

    #tracking mode detection
    if GPIO.input(GPIO_manual_mode)==True:
        cv2.putText(ndarray, 'TRACKING MODE', (1030, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (136, 255, 12), 2)
        cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        minimum_area = 10
        for c in cnts:
            area = cv2.contourArea(c)
            if area > minimum_area:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX, cY), 6, (136,255,12), 2)
                x,y,w,h = cv2.boundingRect(c)
                rows = [x,y]
                with open(filename, 'ab',) as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(rows)
                x_value = x
                cv2.putText(ndarray, 'X value: {}'.format(x), (50,805), cv2.FONT_HERSHEY_SIMPLEX, 1, (136, 255, 12), 2)
                cv2.putText(ndarray, 'Y value: {}'.format(y), (50,855), cv2.FONT_HERSHEY_SIMPLEX, 1, (136, 255, 12), 2)
                break

    #tracking mode speed setting
        if 0 < x_value < 150:
            speed = tic_base_speed +((150-x_value) * tic_diff)
        elif 150 <= x_value < 280:

            speed = tic_base_speed
        else:
            speed = 0
            x_value = 0
        ticcmd('--exit-safe-start', '--velocity', str(int(speed)))

    #manual mode detection
    else:
        cv2.putText(ndarray, 'MANUAL MODE', (1050, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200, 200, 200), 2)
        cnts = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        minimum_area = 10
        for c in cnts:
            area = cv2.contourArea(c)
            if area > minimum_area:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX, cY), 6, (136,255,12), 2)
                x,y,w,h = cv2.boundingRect(c)
                rows = [x,y]
                with open(filename, 'ab',) as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(rows)
                x_value = x
                cv2.putText(ndarray, 'X value: {}'.format(x), (50,805), cv2.FONT_HERSHEY_SIMPLEX, 1, (136, 255, 12), 2)
                cv2.putText(ndarray, 'Y value: {}'.format(y), (50,855), cv2.FONT_HERSHEY_SIMPLEX, 1, (136, 255, 12), 2)
                break
    #manual mode speed setting
        speed = tic_max_speed*pot2.value
        ticcmd('--exit-safe-start', '--velocity', str(int(speed)))

    #adding video streams to ndarray
    frame = imutils.resize(frame, width=1000)
    ndarray[0:750,0:1000] = frame
    ndarray[190:430,1060:1380] = mask1_disp
    ndarray[470:710,1060:1380] = mask2_disp
    #text display on ndarray
    belt_speed = int(speed/170000)
    cv2.putText(ndarray, 'Belt Speed: {} mm/s'.format(belt_speed), (330, 830), cv2.FONT_HERSHEY_SIMPLEX, 1, (96, 55, 212), 2)
    cv2.putText(ndarray, '@evo_biomech', (1190, 870), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Frame", ndarray)
    #saving video stream
    out.write(ndarray)

    #emergency exit
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cv2.destroyAllWindows()
GPIO.cleanup()
ticcmd('--enter-safe-start')
ticcmd('--deenergize')
vs.stop()