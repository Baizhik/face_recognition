# Face Recognition Attendance System

This project is a real-time face recognition attendance system built with Python, OpenCV, and Firebase.  
It detects faces from a webcam, compares them with stored face encodings, and marks attendance automatically for recognized students.

## Features

- Real-time face detection and recognition
- Automatic attendance marking
- Firebase Realtime Database integration
- Student information display on the UI
- Unknown face handling
- Duplicate attendance prevention
- Performance optimization using caching and frame skipping
- Time-based UI transitions
- Multi-pass confirmation for more stable recognition

## Technologies Used

- **Python** – main programming language
- **OpenCV** – webcam access, image processing, and UI rendering
- **face_recognition** – face detection and face encoding
- **Firebase Realtime Database** – storing student data and attendance info
- **cvzone** – drawing styled bounding boxes
- **NumPy** – numerical operations for face matching

## How It Works

1. The webcam captures live video.
2. The system detects faces from the camera frame.
3. Each detected face is converted into an encoding.
4. The encoding is compared with stored known encodings.
5. If the distance is below the threshold, the face is recognized.
6. The same face must be detected multiple times in a row before confirmation.
7. After confirmation, attendance is marked in Firebase.
8. The UI shows student information or an **Unknown** label if no valid match is found.

<img width="1601" height="936" alt="image" src="https://github.com/user-attachments/assets/dba2ce49-a007-41d1-b409-3d2d91af6563" />
