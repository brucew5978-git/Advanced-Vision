# import the opencv library
import cv2
import mediapipe as mp
import time

class handDetector():
    #Input characteristics of hand tracking from hands file
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    #initialize class

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # Results are processed in rgb img

        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            # assumes single hand detected
            for handLandMarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandMarks, self.mpHands.HAND_CONNECTIONS)
                    # Draws hand locations over original img, handLandMarks are dots and "HAND_CONNECTIONS" are lines


        return img


    def findPosition(self, img, handNum=0, draw=True):

        LandmarkList = []

        if self.results.multi_hand_landmarks:

            targetHand = self.results.multi_hand_landmarks[handNum]

            for id, landmark in enumerate(targetHand.landmark):
                # print(id, landmark)
                imgHeight, imgWidth, imgChannel = img.shape
                centerX, centerY = int(landmark.x * imgWidth), int(landmark.y * imgHeight)
                # center is connection center of hand

                #print(id, centerX, centerY)
                LandmarkList.append([id, centerX, centerY])

                if draw:
                    cv2.circle(img, (centerX, centerY), 5, (255, 255, 0), cv2.FILLED)
                # Each hand has 20 ID points, id == 0 is palm point of hand, ==4 is thumb point

            #For target hand, will get all landmarks

        return LandmarkList



def main():
    previousTime = 0
    currentTime = 0
    # Time var for tracking fps

    # define a video capture object
    cap = cv2.VideoCapture(0)

    detector = handDetector()

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

if __name__ == "__main__":
    main()