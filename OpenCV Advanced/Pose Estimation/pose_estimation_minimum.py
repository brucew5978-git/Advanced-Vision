import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('PoseVideos/Video3.mp4')

previousTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Need to convert img into RGB as mp only supports RGB

    results = pose.process(imgRGB)
    #print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            print(id, landmark)
            centerX, centerY = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(img, (centerX, centerY), 5, (255,0,0), cv2.FILLED)
            #Places circles on mediapipe landmark 
            
    #Draw landmarks when landmarks detected
    #"POSE_CONNECTIONS" fills connections btw dots

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (244,0,0), 3)
    #Puts text on screen

    cv2.imshow("Image", img)
    
    cv2.waitKey(10)
    #Increasing number here decreases frame rate - slows video