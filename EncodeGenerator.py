import cv2
import face_recognition
import pickle
import os

import firebase_admin 
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendanceproject-e874c-default-rtdb.firebaseio.com/",
    'storageBucket': "attendanceproject-e874c.appspot.com"
})

folderPathImages = 'Images'
listPathImages = os.listdir(folderPathImages)
imgListImages = []

studentIDs = []
# print(listPathImages)

for path in listPathImages:
    imgListImages.append(cv2.imread(os.path.join(folderPathImages, path)))
    studentIDs.append(os.path.splitext(path)[0]) 

    fileName = f'{folderPathImages}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

def generateEncodings(images):
    encodingList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingList.append(encode)

    return encodingList

encodingListKnown = generateEncodings(imgListImages)
encodingsListWithIDs =  [encodingListKnown, studentIDs]

encodingFile = open("EncodingsFIle.p", "wb")

pickle.dump(encodingsListWithIDs, encodingFile)

encodingFile.close()


