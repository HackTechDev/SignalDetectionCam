import numpy as np
import cv2
import glob


cap = cv2.VideoCapture(0)

while True:
    ret, img_rgb = cap.read()


    cv2.imshow('Detected',img_rgb)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
