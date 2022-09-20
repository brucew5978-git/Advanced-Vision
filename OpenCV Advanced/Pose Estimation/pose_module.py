import cv2
import mediapipe as mp
import time

class poseDetector():

    #creates object from a class
    def __init__(self, mode=False, upBody = 1, enableSeg = False,
                 smoothSeg = True, smooth = True, detectionCon = 0.5, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.enableSeg, self.smoothSeg, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Need to convert img into RGB as mp only supports RGB

        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        landmarkList = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                #print(id, landmark)
                centerX, centerY = int(landmark.x * width), int(landmark.y * height)

                landmarkList.append([id, centerX, centerY])
                if draw:
                    cv2.circle(img, (centerX, centerY), 5, (255,0,0), cv2.FILLED)
                    #Places circles on mediapipe landmark

        return landmarkList
    #Draw landmarks when landmarks detected
    #"POSE_CONNECTIONS" fills connections btw dots

def main():
    cap = cv2.VideoCapture('PoseVideos/Video3.mp4')
    previousTime = 0
    detector = poseDetector()



    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        landmarkList = detector.findPosition(img, draw=False)

        print(landmarkList[14])
        cv2.circle(img, (landmarkList[14][1], landmarkList[14][2]), 15, (0,0,255), cv2.FILLED)
        #Specifically tracks landmark 14 from list

        currentTime = time.time()
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (244,0,0), 3)
        #Puts text on screen

        cv2.imshow("Image", img)

        cv2.waitKey(10)
        #Increasing number here decreases frame rate - slows video


if __name__ == "__main__":
    main()
