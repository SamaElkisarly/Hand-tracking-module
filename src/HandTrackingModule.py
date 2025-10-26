import cv2
import time #to check frame rate
import mediapipe as mp

class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,1,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert it to rgb because this object only uses rgb
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:#responding to flag
                 self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)  # we don't want to draw the rgb image , we don't want to display it but the orig image,we tell him to draw for each hand and show connections(lines between points)
        return img#if we drew on it

    def findposition(self, img, handNo=0, draw=True):
        lmlist =[]  # landmark positiions
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape  # height,width,channel
                cx, cy = int(lm.x * w), int(lm.y * h)  # instead of decimals , we want to see pixels
                # print(id, cx, cy)  # id of each landmark,the x pixel,y pixel
                lmlist.append([id, cx, cy])
                # highlighting a landmark
                # if id == 4:
                if (draw):
                    cv2.circle(img, (cx, cy), 10, (254, 130, 140), cv2.FILLED)
        return lmlist  # whether filled or not


def main():
    #dummy code that you can run in a different project
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # you don't have another camera
    #define obj, we don't need parameters since we already have default params
    detector = HandDetector()
    while True:
        success, img = cap.read()  # we'll stick with default parameters
        img=detector.findHands(img)#it returns an img
        lmlist=detector.findposition(img)
        if len(lmlist) !=0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=="__main__":#if we are running this script
    main()

