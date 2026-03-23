import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-6404a-default-rtdb.firebaseio.com/"
}) 

ref = db.reference('Students') 

data = {
    "230106":
        {
            "name": "Anna de armas",
            "major": "Actress",
            "starting_year": 2017,
            "total_attendance": 3,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2026-03-22 00:54:34"
        },
    "230107":
        {
            "name": "Tom Holland",
            "major": "Management",
            "starting_year": 2021,
            "total_attendance": 4,
            "standing": "B",
            "year": 1,
            "last_attendance_time": "2026-03-22 00:54:34"
        },
    "230108":
        {
            "name": "Baizhan Baubek",
            "major": "Engineering",
            "starting_year": 2023,
            "total_attendance": 7,
            "standing": "G",
            "year": 3,
            "last_attendance_time": "2026-03-22 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)