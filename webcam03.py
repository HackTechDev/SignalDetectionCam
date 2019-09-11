import numpy as np
import cv2
import glob

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Detected")
cv2.createTrackbar("Threshold_tb", "Detected", 1, 10, nothing)


while True:
    ret, img_rgb = cap.read()

    threshold_value = cv2.getTrackbarPos("Threshold_tb", "Detected")


    for f in glob.iglob("signe/*"):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        template = cv2.imread(f, 0)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

        threshold = threshold_value / 10

        loc = np.where( res >= threshold)

        flag = False
        for pt in zip(*loc[::-1]):
            if pt is not None:
                flag = True
                break
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

        if flag == True:
            print(f, end='')
            print("Detected")
            break

        cv2.imshow('Detected',img_rgb)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
