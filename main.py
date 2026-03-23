import cv2
import os
import pickle
import face_recognition
import numpy
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-6404a-default-rtdb.firebaseio.com/"
})

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackgroundBase = cv2.imread('resources/background.png')

folderModePath = 'resources/modes'
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

with open("EncodeFile.p", 'rb') as file:
    encodeListKnownWithId = pickle.load(file)

encodeListKnown, studentsIds = encodeListKnownWithId

studentImageCache = {}
for student_id in studentsIds:
    img_path = f"images/{student_id}.jpg"
    if os.path.exists(img_path):
        studentImageCache[str(student_id)] = cv2.imread(img_path)

studentInfoCache = db.reference('Students').get()
if studentInfoCache is None:
    studentInfoCache = {}

modeType = 0
counter = 0
id = -1

studentInfo = None
imgStudent = None

frame_count = 0
process_every_n_frames = 2

stateStartTime = None
infoDisplaySeconds = 5
doneDisplaySeconds = 1
duplicateDisplaySeconds = 2

last_faceCurFrame = []
last_encodeCurFrame = []

match_threshold = 0.5
last_unknown_bbox = None
unknownDisplayFrames = 8
unknownCounter = 0

pendingId = None
stableMatchCount = 0
requiredStableMatches = 3


def reset_stable_match():
    global pendingId, stableMatchCount
    pendingId = None
    stableMatchCount = 0


while True:
    success, img = cap.read()
    if not success:
        continue

    frame_count += 1
    should_process = (frame_count % process_every_n_frames == 0)

    if should_process:
        current_unknown_bbox = None

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        last_faceCurFrame = face_recognition.face_locations(imgS)
        last_encodeCurFrame = face_recognition.face_encodings(imgS, last_faceCurFrame)

    imgBckgrnd = imgBackgroundBase.copy()
    imgBckgrnd[162:162 + 480, 55:55 + 640] = img
    imgBckgrnd[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if unknownCounter > 0 and last_unknown_bbox is not None and counter == 0:
        imgBckgrnd = cvzone.cornerRect(imgBckgrnd, last_unknown_bbox, rt=0)

        x, y, w, h = last_unknown_bbox
        text_y = y - 10 if y - 10 > 20 else y + h + 25

        cv2.putText(imgBckgrnd, "Unknown", (x, text_y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        unknownCounter -= 1

    if last_faceCurFrame:
        recognized_this_cycle = False
        confirmed_this_cycle = False

        for encodeFace, FaceLoc in zip(last_encodeCurFrame, last_faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchId = numpy.argmin(faceDist)
            best_distance = faceDist[matchId]

            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1

            if matches[matchId] and best_distance < match_threshold:
                recognized_this_cycle = True
                last_unknown_bbox = None
                unknownCounter = 0

                imgBckgrnd = cvzone.cornerRect(imgBckgrnd, bbox, rt=0)

                currentMatchedId = studentsIds[matchId]

                if counter == 0:
                    if currentMatchedId == pendingId:
                        stableMatchCount += 1
                    else:
                        pendingId = currentMatchedId
                        stableMatchCount = 1

                    if stableMatchCount >= requiredStableMatches:
                        id = currentMatchedId
                        counter = 1
                        modeType = 1
                        stateStartTime = datetime.now()
                        confirmed_this_cycle = True
                        reset_stable_match()
                else:
                    confirmed_this_cycle = True

                break

            else:
                current_unknown_bbox = bbox

        if should_process and not recognized_this_cycle and current_unknown_bbox is not None and counter == 0:
            last_unknown_bbox = current_unknown_bbox
            unknownCounter = unknownDisplayFrames
            reset_stable_match()

        if should_process and not recognized_this_cycle and counter == 0:
            reset_stable_match()

        if counter != 0:

            if counter == 1:
                studentInfo = studentInfoCache.get(str(id))
                imgStudent = studentImageCache.get(str(id))

                if studentInfo is None:
                    counter = 0
                    modeType = 0
                    stateStartTime = None
                    last_faceCurFrame = []
                    last_encodeCurFrame = []
                    reset_stable_match()
                    continue

                datetimeObject = datetime.strptime(
                    studentInfo['last_attendance_time'],
                    "%Y-%m-%d %H:%M:%S"
                )
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()

                if secondsElapsed > 30:
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ref = db.reference(f'Students/{id}')

                    studentInfo['total_attendance'] += 1
                    studentInfo['last_attendance_time'] = now_str

                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(now_str)

                    modeType = 1
                    stateStartTime = datetime.now()
                    counter = 2

                else:
                    modeType = 3
                    stateStartTime = datetime.now()
                    counter = 2

            if modeType != 3:
                elapsed = (datetime.now() - stateStartTime).total_seconds()

                if elapsed <= infoDisplaySeconds:
                    modeType = 1
                    imgBckgrnd[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    cv2.putText(imgBckgrnd, str(studentInfo['total_attendance']), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBckgrnd, str(studentInfo['major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBckgrnd, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBckgrnd, str(studentInfo['standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBckgrnd, str(studentInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBckgrnd, str(studentInfo['starting_year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBckgrnd, str(studentInfo['name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    if imgStudent is not None:
                        imgBckgrnd[175:175 + 216, 909:909 + 216] = imgStudent

                elif elapsed <= infoDisplaySeconds + doneDisplaySeconds:
                    modeType = 2
                    imgBckgrnd[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                else:
                    counter = 0
                    modeType = 0
                    id = -1
                    studentInfo = None
                    imgStudent = None
                    stateStartTime = None
                    last_faceCurFrame = []
                    last_encodeCurFrame = []
                    reset_stable_match()

            else:
                imgBckgrnd[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                elapsed = (datetime.now() - stateStartTime).total_seconds()

                if elapsed > duplicateDisplaySeconds:
                    counter = 0
                    modeType = 0
                    id = -1
                    studentInfo = None
                    imgStudent = None
                    stateStartTime = None
                    last_faceCurFrame = []
                    last_encodeCurFrame = []
                    reset_stable_match()

    else:
        if should_process:
            modeType = 0
            counter = 0
            id = -1
            studentInfo = None
            imgStudent = None
            stateStartTime = None
            last_faceCurFrame = []
            last_encodeCurFrame = []
            last_unknown_bbox = None
            unknownCounter = 0
            reset_stable_match()

    cv2.imshow("Attendance", imgBckgrnd)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows() 



# 1) Perfomance out of loop 
# 2) Not frame but time based ui 
# 3) Threshold for distance and the frame n pass 
# 4) Multipass - so it should be detected several times 
# 5) Unknown GUI