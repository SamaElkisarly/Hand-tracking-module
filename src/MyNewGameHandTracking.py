import cv2
import time #to check frame rate
import mediapipe as mp
import HandTrackingModule as htm
# dummy code that you can run in a different project
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)  # you don't have another camera
# define obj, we don't need parameters since we already have default params
detector = htm.HandDetector()
while True:
    success, img = cap.read()  # we'll stick with default parameters
    img = detector.findHands(img,draw=False)  # it returns an img #(img)if you want to show drawing
    lmlist = detector.findposition(img,draw=False) #(img)if you want to show drawing
    if len(lmlist) != 0:
        print(lmlist[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
