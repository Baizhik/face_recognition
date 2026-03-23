import cv2
import face_recognition
import pickle 
import os


# import students 
folderModePath = 'images'
modePathList = os.listdir(folderModePath)
imgList = []
studentsId = []


for path in modePathList:
    imgList.append(cv2.imread(os.path.join(folderModePath,path))) 
    studentsId.append(os.path.splitext(path)[0])

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

encodeListKnown = findEncodings(imgList)
encodeListKnownWithId = [encodeListKnown,studentsId]

file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithId,file)
file.close()
