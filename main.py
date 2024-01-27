import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone

import firebase_admin 
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendanceproject-e874c-default-rtdb.firebaseio.com/",
    'storageBucket': "attendanceproject-e874c.appspot.com/"
})

bucket = storage.bucket()

# initializing webcam and reading background image
capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)
backgroundImg = cv2.imread('Resources/Background.png')

folderPathMode = 'Resources/Modes'
listPathMode = os.listdir(folderPathMode)
imgListMode = []

for path in listPathMode:
    imgListMode.append(cv2.imread(os.path.join(folderPathMode, path)))

encodingsFile = open('EncodingsFile.p', "rb")
encodingListWithIDs = pickle.load(encodingsFile)
encodingsFile.close()

encodingsListKnown, studentIDs = encodingListWithIDs
print(studentIDs)

modeType = 0
counter = 0
id = -1

while True:
    success, image = capture.read()

    # Resize the webcam image to match the region size
    image = cv2.resize(image, (640, 480))

    # Assign the resized image to the background
    backgroundImg[162:162 + 480, 55:55 + 640] = image
    backgroundImg[44:44 + 633, 808:808 + 414] = imgListMode[0]

    smallImage = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    smallImage = cv2.cvtColor(smallImage, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(smallImage)

    encodeCurrentFrame = face_recognition.face_encodings(smallImage, faceCurrentFrame)
    # showing the window for your backgrounding
    cv2.imshow("Attendance System", backgroundImg)
    cv2.waitKey(1)

    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodingsListKnown, encodeFace, tolerance=0.6)
        faceDistance = face_recognition.face_distance(encodingsListKnown, encodeFace)
        print("Matches", matches)
        print("Face Distance", faceDistance)

        matchIndex = np.argmin(faceDistance)
        print("Match Index", matchIndex)

        if matches[matchIndex]:
            print("Registered Student Detected")
            print(studentIDs[matchIndex])

        y1, x2, x1, y2 = faceLocation
        y1, x2, x1, y2 = y1 * 4, x2 * 4, x1 * 4, y2 * 4
        bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1

        backgroundImg = cvzone.cornerRect(backgroundImg, bbox, rt=0)

    cv2.imshow("Attendance System", backgroundImg)
    cv2.waitKey(1)