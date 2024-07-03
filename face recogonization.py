import math
import cv2
import cvzone
from ultralytics import YOLO

import os
import requests
import face_recognition
import numpy as np
from datetime import datetime
import pyttsx3

engine = pyttsx3.init()
firebase_url = "https://face-attendance-36b4a-default-rtdb.firebaseio.com/"

def update_attendance(name):
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Check if attendance for the person already exists for the current date
    response = requests.get(firebase_url + f"attendance/{current_date}.json")
    if response.status_code == 200:
        data = response.json()
        if data:
            for entry in data.values():
                if entry['name'] == name:
                    text = f"Attendance for {name} already uploaded for today."
                    print(text)
                    engine.say(text)
                    engine.runAndWait()
                    return 0
    
    # If attendance doesn't exist, add the record to Firebase
    data = {"name": name, "time": current_time}
    response = requests.post(firebase_url + f"attendance/{current_date}.json", json=data)
    if response.status_code == 200:
        print(f"Attendance for {name} uploaded successfully.")
        text = f"Attendance for {name} uploaded successfully."
        print(text)
        engine.say(text)
        engine.runAndWait()
        return 1
    else:
        print(f"Failed to upload attendance for {name}.")
        text = f"Failed to upload attendance for {name}."
        print(text)
    engine.say(text)
    engine.runAndWait()
    return 0

def check_attendance_firebase(name):
    current_date = datetime.now().strftime("%Y-%m-%d")
    response = requests.get(firebase_url + f"attendance/{current_date}.json")
    if response.status_code == 200:
        data = response.json()
        if data:
            for entry in data.values():
                if entry['name'] == name:
                    return True
    return False

# Set properties (optional)
rate = engine.getProperty('rate')  # Get current speaking rate
engine.setProperty('rate', rate - 50)  # Decrease the rate
volume = engine.getProperty('volume')  # Get current volume level (0.0 to 1.0)
engine.setProperty('volume', volume + 0.25)  # Increase volume

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Change to a different voice if needed

# Load known faces and initialize lists
known_face_encodings = []
known_face_names = []
students_folder = 'students_photos'
for filename in os.listdir(students_folder):
    student_name = os.path.splitext(filename)[0]  # Extract the student's name from the filename
    student_image = face_recognition.load_image_file(os.path.join(students_folder, filename))
    student_encoding = face_recognition.face_encodings(student_image)[0]
    known_face_encodings.append(student_encoding)
    known_face_names.append(student_name)

confidence = 0.3
cap = cv2.VideoCapture(0)  # For Webcam

# Initialize the YOLO model to run on CPU
model = YOLO("model/mytrain2.pt")  # Adding device='cpu' to ensure it runs on CPU

classNames = ["fake", "real"]
prev_frame_time = 0
new_frame_time = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if conf > confidence:
                if classNames[cls] == 'real':
                    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    confidence_threshold = 0.4

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index] and face_distances[best_match_index] > confidence_threshold:
                            name = known_face_names[best_match_index]
                            attendance_marked = check_attendance_firebase(name)
                            if attendance_marked:
                                present_label = "Present"
                            else:
                                present_label = ""
                            cv2.putText(img, present_label, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                            if update_attendance(name):
                                cvzone.putTextRect(img, f'{name}', (left * 4, top * 4 - 10), scale=1.5, thickness=2, colorR=(0, 255, 0), colorB=(255, 255, 255))
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
