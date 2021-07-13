from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time
import cv2

# initialize the camera and stream
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))
stream = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
#Doesnt seem to work without this line
camera.close()
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()
# loop over some frames...this time using the threaded stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# update the FPS counter
	fps.update()
    if cv2.waitKey(1):
        print("Stopping")
        break
# stop the timer and display FPS information
fps.stop()
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()