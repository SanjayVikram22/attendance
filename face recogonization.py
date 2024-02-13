import os
import requests
import face_recognition
import cv2
import numpy as np
from datetime import datetime

# Initialize Firebase URL
firebase_url = "https://face-attendance-36b4a-default-rtdb.firebaseio.com/"

# Function to update attendance in Firebase
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
                    print(f"Attendance for {name} already uploaded for today.")
                    return
    
    # If attendance doesn't exist, add the record to Firebase
    data = {"name": name, "time": current_time}
    response = requests.post(firebase_url + f"attendance/{current_date}.json", json=data)
    if response.status_code == 200:
        print(f"Attendance for {name} uploaded successfully.")
    else:
        print(f"Failed to upload attendance for {name}.")

# Load known faces
known_face_encodings = []
known_face_names = []

# Load photos from the 'students_photos' folder
students_folder = 'students_photos'
for filename in os.listdir(students_folder):
    student_name = os.path.splitext(filename)[0]  # Extract the student's name from the filename
    student_image = face_recognition.load_image_file(os.path.join(students_folder, filename))
    student_encoding = face_recognition.face_encodings(student_image)[0]
    known_face_encodings.append(student_encoding)
    known_face_names.append(student_name)

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    """for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # Update attendance in Firebase
            update_attendance(name)"""
    
    
    # Set the confidence threshold
    confidence_threshold = 0.4

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding) # Adjust tolerance as per your requirement
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        print(face_distances[best_match_index])

        if matches[best_match_index] and face_distances[best_match_index] > confidence_threshold:
            name = known_face_names[best_match_index]
            # Update attendance in Firebase
            update_attendance(name)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
