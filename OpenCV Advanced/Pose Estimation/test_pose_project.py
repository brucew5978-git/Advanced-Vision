import cv2
import time
import pose_module as pm

cap = cv2.VideoCapture('PoseVideos/Video3.mp4')
previousTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    landmarkList = detector.findPosition(img, draw=False)

    targetLandmark = 16
    print(landmarkList[targetLandmark])
    cv2.circle(img, (landmarkList[targetLandmark][1], landmarkList[targetLandmark][2]), 15, (0,0,255), cv2.FILLED)
    #Specifically tracks landmark 14 from list

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (244,0,0), 3)
    #Puts text on screen

    cv2.imshow("Image", img)

    cv2.waitKey(10)
    #Increasing number here decreases frame rate - slows video
