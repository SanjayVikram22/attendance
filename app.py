from flask import Flask, render_template, request
import requests
from datetime import datetime

app = Flask(__name__)

# Initialize Firebase URL
firebase_url = "https://face-attendance-36b4a-default-rtdb.firebaseio.com/"

# Function to fetch attendance data for a specific date
def fetch_attendance(date):
    response = requests.get(firebase_url + f"attendance/{date}.json")
    if response.status_code == 200:
        return response.json()
    return None

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get selected date from the form
        selected_date = request.form['date']
        # Fetch attendance data for the selected date
        attendance_data = fetch_attendance(selected_date)
        if attendance_data:
            return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)
        else:
            return render_template('attendance_not_available.html', selected_date=selected_date)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
