# import the necessary packages
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2 # OpenCV library
import numpy as np
import subprocess
def ticcmd(*args):
    return subprocess.check_output(['ticcmd'] + list(args))

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
    frame = frame.array

    mask = np.ones(frame.shape, dtype=np.uint8)
    mask.fill(255)
    roi_corners = np.array([[(0, -10), (0, 480), (620, 350), (620,140)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 0)
    cut_frame = cv2.bitwise_or(frame, mask)

    grayscaled = cv2.cvtColor(cut_frame, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(grayscaled, 80, 255, cv2.THRESH_BINARY_INV)

    cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    minimum_area = 50
    maximum_area = 1000
    for c in cnts:

        area = cv2.contourArea(c)
        if maximum_area > area > minimum_area:
            # Find centroid
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 20, (36, 255, 12), 2)
            x, y, w, h = cv2.boundingRect(c)
            x_value = x
            cv2.putText(frame, 'X value: {}'.format(x), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            cv2.putText(frame, 'Y value: {}'.format(y), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            break

    # Display the frame using OpenCV
    cv2.imshow("Frame", frame)
    #cv2.imshow("Cut Frame", cut_frame)
    #cv2.imshow("Threshold", threshold)
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)

    if 0 < x_value < 300:
        speed = base_speed +((300-x_value) * diff)

    elif 300 <= x_value < 550:
        speed = base_speed
    else:
        speed = 0

    ticcmd('--energize')
    ticcmd('--exit-safe-start', '--velocity', str(speed))



    # If the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

ticcmd('--enter-safe-start')
ticcmd('--deenergize')
cv2.destroyAllWindows()