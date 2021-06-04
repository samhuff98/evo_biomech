import cv2

cam = cv2.VideoCapture(1)
focus = 0  # min: 0, max: 255, increment:5
cam.set(28, focus)

while True:
    ret, frame = cam.read()
    cv2.imshow('cam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

cam.release()
cv2.destroyAllWindows()

