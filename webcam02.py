import cv2
import glob
import numpy as np


for f in glob.iglob("signe/*"):
    img_rgb = cv2.imread('telegraphe/IMG_20190909_192741960.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    print(f)
    template = cv2.imread(f, 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

    cv2.imshow('Detected',img_rgb)
    cv2.waitKey(0)
