from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
import time
import datetime
import requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

app = Flask(__name__)

# تهيئة Airtable
AIRTABLE_BASE_ID = "appIxTUTsazkHkVid"
AIRTABLE_TABLE_NAME_ATTENDANCE = "Attendance"
AIRTABLE_TABLE_NAME_UNKNOWN = "unknown"
AIRTABLE_API_KEY = "Bearer patHvtSvk5NHMr8x3.f7f7c265e6fdcef06935bdfe270f2d94762fa08193fef45993f1717b72d6ffcf"
AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}"

# تهيئة Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']
creds = None

def authenticate_gdrive():
    global creds
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

def get_drive_folder_id(folder_name="FaceImages"):
    try:
        authenticate_gdrive()
        drive_service = build('drive', 'v3', credentials=creds)
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        if items:
            return items[0]['id']
        else:
            file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
            folder = drive_service.files().create(body=file_metadata, fields='id').execute()
            return folder['id']
    except Exception as e:
        print(f"Error getting or creating Google Drive folder ID: {e}")
        return None

def upload_to_drive(file_path, folder_id):
    try:
        authenticate_gdrive()
        drive_service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': os.path.basename(file_path), 'parents': [folder_id]}
        media = MediaFileUpload(file_path, mimetype='image/jpeg')
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        file_url = f"https://drive.google.com/file/d/{file['id']}/view?usp=sharing"
        return file_url
    except Exception as e:
        print(f"Error uploading to Google Drive: {e}")
        return None

class FaceRecognizer:
    def __init__(self):
        self.face_encodings_known_list = []
        self.face_name_known_list = []
        self.load_known_faces()

    def load_known_faces(self):
        if os.path.exists("features_airtable.csv"):
            csv_rd = pd.read_csv("features_airtable.csv", header=None)
            for i in range(csv_rd.shape[0]):
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_encodings_known_list.append([float(x) for x in csv_rd.iloc[i][1:129]])
            print("Known faces loaded:", len(self.face_encodings_known_list))
        else:
            print("'features_airtable.csv' not found!")

    def recognize_face(self, frame, folder_id):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "unknown"
            color = (0, 0, 255)  # Red for unknown

            matches = face_recognition.compare_faces(self.face_encodings_known_list, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = self.face_name_known_list[first_match_index]
                color = (0, 255, 0)  # Green for known
                self.send_attendance_to_airtable(name)

            results.append({
                "name": name,
                "color": color,
                "box": [left, top, right, bottom]
            })

            if name == "unknown":
                self.send_unknown_to_airtable_with_image(frame, (top, right, bottom, left), folder_id)

        return results

    def send_attendance_to_airtable(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        data = {
            "fields": {
                "Name": name,
                "Time": current_time,
                "Date": current_date
            }
        }
        headers = {
            "Authorization": AIRTABLE_API_KEY,
            "Content-Type": "application/json"
        }
        response = requests.post(f"{AIRTABLE_URL}/{AIRTABLE_TABLE_NAME_ATTENDANCE}", json=data, headers=headers)
        if response.status_code == 201:
            print(f"{name} marked as present for {current_date} at {current_time} in Airtable")
        else:
            print(f"Error: {response.status_code}, Could not add attendance for {name}")

    def send_unknown_to_airtable_with_image(self, img_rd, face_location, folder_id):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H-%M-%S')
        file_name = f"unknown_{current_date}_{current_time}.jpg"

        # حفظ الصورة مؤقتًا
        temp_image_path = os.path.join("temp_unknown_faces", file_name)
        if not os.path.exists("temp_unknown_faces"):
            os.makedirs("temp_unknown_faces")
        top, right, bottom, left = face_location
        cropped_face = img_rd[top:bottom, left:right]
        cv2.imwrite(temp_image_path, cropped_face)

        # تحميل الصورة إلى Google Drive
        drive_url = upload_to_drive(temp_image_path, folder_id)
        if drive_url:
            # إرسال البيانات إلى Airtable
            upload_data = {
                "fields": {
                    "Name": "unknown",
                    "Time": current_time,
                    "Date": current_date,
                    "Status": "Unauthorized",
                    "Image": drive_url  # إرسال الرابط كـ نص عادي
                }
            }
            headers = {
                "Authorization": AIRTABLE_API_KEY,
                "Content-Type": "application/json"
            }
            response = requests.post(f"{AIRTABLE_URL}/{AIRTABLE_TABLE_NAME_UNKNOWN}", json=upload_data, headers=headers)
            if response.status_code == 201:
                print(f"Unknown face recorded at {current_date} {current_time} in the 'unknown' table.")
            else:
                print(f"Error: {response.status_code}, Could not record unknown face. Response: {response.text}")

        # حذف الصورة المؤقتة
        os.remove(temp_image_path)

face_recognizer = FaceRecognizer()

def generate_frames():
    folder_id = get_drive_folder_id()
    if not folder_id:
        print("Error: Unable to find Google Drive folder.")
        return

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # تغيير حجم الإطار إلى 640x480 لتقليل استخدام الذاكرة
            frame = cv2.resize(frame, (640, 480))

            # الكشف عن الوجوه
            results = face_recognizer.recognize_face(frame, folder_id)
            for result in results:
                x1, y1, x2, y2 = result['box']
                color = result['color']
                name = result['name']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
