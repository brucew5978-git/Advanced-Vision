#imports features in "handtracking_module" so can use various projects outside of file

import cv2
import mediapipe as mp
import time
import handtracking_module as htm

previousTime = 0
currentTime = 0
# Time var for tracking fps

# define a video capture object
cap = cv2.VideoCapture(0)

detector = htm.handDetector()

while (True):
    # Capture the video frame by frame
    success, img = cap.read()
    img = detector.findHands(img)

    newLandmarkList = []
    newLandmarkList = detector.findPosition(img)

    if len(newLandmarkList) != 0:
        print(newLandmarkList[8])
        #Landmark ID 8 is pointer finger, but prints only if length of list is not 0

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)
    # 3 is scale of text and 2 is thickness

    # Display the resulting frame
    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
