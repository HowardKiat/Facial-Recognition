import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://attendanceproject-e874c-default-rtdb.firebaseio.com/"
})

ref = db.reference("Students")

data = {
    "TP011111":
    {
        "name" : "Lee Wen Han",
        "major" : "CS(AI)",
        "starting_year" : 20,
        "grades" : "A",
        "year" : 2,
        "last_attendance_taken" : "2024-01-26 16:10:30",
    },
}

for key, value in data.items():
    ref.child(key).set(value)


