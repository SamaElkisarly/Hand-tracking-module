import cv2
import time #to check frame rate
import mediapipe as mp

cap=cv2.VideoCapture(0)#you don't have another camera
#create an object from our class hands
mpHands=mp.solutions.hands#formality before using model
hands=mpHands.Hands()#object
#draw the 21 points
mpDraw=mp.solutions.drawing_utils
#for frame rate
pTime=0
cTime=0
while True:
    success,img=cap.read()#we'll stick with default parameters
    #send rgb image to that object
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#convert it to rgb because this object only uses rgb
    results=hands.process(imgRGB)
    #open obj and extract info
    #check if we have multiple hands
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
       for handLms in results.multi_hand_landmarks:
           for id,lm in enumerate(handLms.landmark):
               #print(id,lm)
               h,w,c=img.shape#height,width,channel
               cx,cy=int(lm.x*w),int(lm.y*h)#instead of decimals , we want to see pixels
               print(id,cx,cy)#id of each landmark,the x pixel,y pixel
               #highlighting a landmark
               #if id==4:
                #cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

           mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)#we don't want to draw the rgb image , we don't want to display it but the orig image,we tell him to draw for each hand and show connections(lines between points)
#extract info of each hand
#current time(cT) :time of catching the new frame
    #previous time(pt):time of catching past frame
    #frame per second(fps): number of frames caught between ct and pt
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)