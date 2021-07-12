# import the necessary packages
import argparse
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2 # OpenCV library
import numpy as np
import subprocess
from evo_biomech import PyVideoStream

def ticcmd(*args):
    return subprocess.check_output(['ticcmd'] + list(args))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,help="Whether or not frames should be displayed")
args = vars(ap.parse_args())
# initialize the camera and stream
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))
stream = camera.capture_continuous(rawCapture, format="bgr",use_video_port=True)

# created a *threaded *video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from `picamera` module...")
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()
# loop over some frames...this time using the threaded stream
while fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	# update the FPS counter
	fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

ticcmd('--reset')
ticcmd('--step-mode', str(int(4)))
ticcmd('--max-accel', str(int(1500000)))
ticcmd('--max-decel', str(int(1500000)))
ticcmd('--max-speed', str(int(18000000)))
ticcmd('--current', str(int(320)))

base_speed = 5000000
diff = 20000

# Initialize the camera
camera = PiCamera()

camera.start_preview()

# Set the camera resolution
camera.resolution = (640, 480)

# Set the number of frames per second
camera.framerate = 32

# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(640, 480))

# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)

# Capture frames continuously from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

    x_value = 0
    speed = 0

    # Grab the raw NumPy array representing the image
    image = frame.array

    #mask = np.ones(frame.shape, dtype=np.uint8)
    #mask.fill(255)
    #corners = [(20, 10), (35, 455), (612, 312), (615,135)]
    #roi_corners = np.array([corners], dtype=np.int32)
    #cv2.fillPoly(mask, roi_corners, 0)
    #cut_frame = cv2.bitwise_or(frame, mask)

    #cv2.line(frame, corners[0], corners[1], (136,55,212), 1)
    #cv2.line(frame, corners[1], corners[2], (136,55,212), 1)
    #cv2.line(frame, corners[2], corners[3], (136,55,212), 1)
    #cv2.line(frame, corners[3], corners[0], (136,55,212), 1)

    #grayscaled = cv2.cvtColor(cut_frame, cv2.COLOR_BGR2GRAY)
    #retval, threshold = cv2.threshold(grayscaled, 50, 255, cv2.THRESH_BINARY_INV)

    #cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    #minimum_area = 300
    #maximum_area = 1000
    #for c in cnts:

    #    area = cv2.contourArea(c)
    #    if  area > minimum_area:
    #        # Find centroid
    #        M = cv2.moments(c)
    #        cX = int(M["m10"] / M["m00"])
    #        cY = int(M["m01"] / M["m00"])
    #        cv2.circle(frame, (cX, cY), 10, (36, 255, 12), 2)
    #        x, y, w, h = cv2.boundingRect(c)
    #        x_value = x
    #        cv2.putText(frame, 'X value: {}'.format(x), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    #        cv2.putText(frame, 'Y value: {}'.format(y), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    #        break


    #if 0 < x_value < 300:
    #    speed = base_speed +((300-x_value) * diff)

    #elif 300 <= x_value < 550:
    #    speed = base_speed
    #else:
    #    speed = 0
    #    x_value = 0

    #ticcmd('--energize')
    #ticcmd('--exit-safe-start', '--velocity', str(speed))
    #cv2.putText(frame, 'Belt Speed: {} mm/s'.format(speed/42500), (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (96, 55, 212), 2)

    # Display the frame using OpenCV
    #cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.resizeWindow("Frame", 10, 10)
    cv2.imshow("Frame", image)
    #cv2.imshow("Cut Frame", cut_frame)
    #cv2.imshow("Threshold", threshold)
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)


    # If the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

ticcmd('--enter-safe-start')
ticcmd('--deenergize')
cv2.destroyAllWindows()