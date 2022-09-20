# import the opencv library
import cv2
import mediapipe as mp
import time


# define a video capture object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
#hands object for processing/eval hands img

mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0
#Time var for tracking fps

while(True):

    # Capture the video frame
    # by frame
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #Results are processed in rgb img

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
    #assumes single hand detected
        for handLandMarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLandMarks.landmark):
                #print(id, landmark)
                imgHeight, imgWidth, imgChannel = img.shape
                centerX, centerY = int(landmark.x * imgWidth), int(landmark.y * imgHeight)
                #center is connection center of hand

                print(id, centerX, centerY)

                if id == 0:
                    cv2.circle(img, (centerX, centerY), 25, (0,255,255), cv2.FILLED)
                #Each hand has 20 ID points, id == 0 is palm point of hand, ==4 is thumb point


            mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)
            #Draws hand locations over original img, handLandMarks are dots and "HAND_CONNECTIONS" are lines

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 2)
    #3 is scale of text and 2 is thickness

    # Display the resulting frame
    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break