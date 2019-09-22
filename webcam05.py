#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# https://ressources.labomedia.org/detection_de_mouvement_avec_opencv_en_python


import numpy as np
import cv2
import glob

def nothing(x):
    pass


def motion_detector(cam, flou, seuil_0, seuil_1, area, tempo):
    # Création des fenêtres
    cv2.namedWindow('Gap_between_frame')
    cv2.namedWindow('Thresh')
    cv2.namedWindow('Security_Feed')
    
    cv2.createTrackbar("Threshold_tb", "Security_Feed", 1, 10, nothing)

    firstFrame = None
    cap = cv2.VideoCapture(cam)
    loop = 1
    counter = 0
    while loop:
        rval, frame = cap.read()

        threshold_value = cv2.getTrackbarPos("Threshold_tb", "Security_Feed")


        # Si la webcam à une image
        if rval:
            # Conversion en gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Application d'un flou
            gray = cv2.GaussianBlur(gray, (flou, flou), 0)

            # Enregistrement d'une 1ère frame
            if firstFrame is None:
                firstFrame = gray
            else:
                # Gap entre les frames
                delta = cv2.absdiff(firstFrame, gray)
                # Seuil
                thresh = cv2.threshold(delta, seuil_0, seuil_1, cv2.THRESH_BINARY)[1]
                # Dilatation des zones
                thresh = cv2.dilate(thresh, None, iterations=2)
                # Contours des zones
                cnts, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                             cv2.CHAIN_APPROX_SIMPLE)
                # Affichage des contours
                for c in contours:
                    if cv2.contourArea(c) > area:

                        print("Mouvement detection: ", counter)
                        firstFrame = None 


                        for f in glob.iglob("signe/*"):
                            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            
                            template = cv2.imread(f, 0)
                            w, h = template.shape[::-1]

                            res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

                            #threshold = threshold_value / 10
                            threshold = 0.8

                            loc = np.where( res >= threshold)

                            flag = False
                            for pt in zip(*loc[::-1]):
                                if pt is not None:
                                    flag = True
                                    break
                                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)

                            if flag == True:
                                print(f, end='')
                                print("Detected")
                                break
                                            


                        # Un rectangle incluant la zone
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        cv2.imshow("Gap_between_frame", delta)
                        cv2.imshow("Thresh", thresh)
                        cv2.imshow("Security_Feed", frame)



                        # Echap pour quitter dans une fenêtre
                        if cv2.waitKey(tempo) & 0xFF == 27:
                            loop = 0

        counter = counter + 1


    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Numero de webcam
    CAM = 0
    
    # Valeur de flou, impair
    FLOU = 41

    # Seuils sur le gris
    SEUIL_0, SEUIL_1 = 60, 255

    # Aire minimal avec différence de pixels
    AREA = 5000
    
    # Attente en ms entre 2 capture
    TEMPO = 30
    
    motion_detector(CAM, FLOU, SEUIL_0, SEUIL_1, AREA, TEMPO)
