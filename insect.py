import cv2
import numpy as np
import subprocess

cap = cv2.VideoCapture(1)
cap.set(28, 20)

# - Initialization -------------------------------------------

def ticcmd(*args):
    return subprocess.check_output(['ticcmd'] + list(args))

ticcmd('--reset')
ticcmd('--step-mode', str(int(4)))
ticcmd('--max-accel', str(int(1500000)))
ticcmd('--max-decel', str(int(1500000)))
ticcmd('--max-speed', str(int(18000000)))
ticcmd('--current', str(int(320)))

base_speed = 6000000
diff = 20000

#set target area
TA = [110, 500, 225, 365]
TA_centre = [((TA[1]-TA[0])/2), ((TA[3]-TA[2])/2)]
print(TA_centre)

while True:
    x_value = 0
    speed = 0

    ret, frame = cap.read()
    #grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #retval, threshold = cv2.threshold(grayscaled, 120, 255, cv2.THRESH_BINARY)
    #res = cv2.bitwise_and(frame, frame, mask=threshold)

    #lower_insect = np.array([100, 100, 100])
    #upper_insect = np.array([200, 200, 200])
    #hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #mask1 = cv2.inRange(hsv1, lower_insect, upper_insect)

    mask1 = np.ones(frame.shape, dtype=np.uint8)
    mask1.fill(255)
    roi_corners = np.array([[(100, -40), (100, 480), (610, 330), (610,150)]], dtype=np.int32)
    cv2.fillPoly(mask1, roi_corners, 0)
    cut_frame = cv2.bitwise_or(frame, mask1)

    grayscaled = cv2.cvtColor(cut_frame, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(grayscaled, 40, 255, cv2.THRESH_BINARY_INV)

    #lower_insect = np.array([30, 20, 20])
    #upper_insect = np.array([200, 200, 200])

    #hsv = cv2.cvtColor(cut_frame, cv2.COLOR_BGR2HSV)
    #mask = cv2.inRange(hsv, lower_insect, upper_insect)
    #res = cv2.bitwise_and(cut_frame, cut_frame, mask=mask)
    #retval, threshold = cv2.threshold(grayscaled, 120, 255, cv2.THRESH_BINARY)
    #cropped = cv2.bitwise_and(frame, frame, mask=mask1)
    #crop_frame = frame[TA[2]:TA[3], TA[0]:TA[1]]
    #hsv = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2HSV)
    #output = cv2.rectangle(frame, (TA[0], TA[2]), (TA[1], TA[3]), (255, 0, 0), 2)

    #lower_red = np.array([30, 30, 130])
    #upper_red = np.array([105, 255, 255])

    #mask = cv2.inRange(hsv, lower_red, upper_red)
    #res = cv2.bitwise_and(crop_frame, crop_frame, mask=mask)


    #invert = cv2.bitwise_not(crop_frame)
    #gray = cv2.cvtColor(invert, cv2.COLOR_BGR2GRAY)
    #blur = cv2.medianBlur(gray, 25)
    #threshold, res = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    #kernel = np.ones((5,5), np.uint8)
    #erosion = cv2.erode(mask, kernel, iterations=1)
    #dilation = cv2.dilate(erosion, kernel, iterations = 1)

    cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    minimum_area = 1000
    for c in cnts:

        area = cv2.contourArea(c)
        if (area > minimum_area):
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

    if 100 < x_value < 300:
        speed = base_speed +((300-x_value) * diff)

    elif 300<= x_value < 550:
        speed = base_speed
    else:
        speed = 0

    ticcmd('--energize')
    ticcmd('--exit-safe-start', '--velocity', str(speed))

    cv2.imshow('frame', frame)
    #cv2.imshow('hsv', hsv)
    #cv2.imshow('mask', mask)
    cv2.imshow('threshold', threshold)
    cv2.imshow('cut_frame', cut_frame)
    #cv2.imshow('dilation', dilation)
    #cv2.imshow('blur', blur)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ticcmd('--enter-safe-start')
ticcmd('--deenergize')
cap.release()
cv2.destroyAllWindows()


