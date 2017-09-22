import subprocess
import cv2
from datetime import datetime
import numpy as np
from requests import request, post

from cena.recognition import FaceRecognizer
from cena.settings import RYAN_SONG_PATH, DEV, CASCADE_FILE_PATH, SERVER_URL


def play_mp3(path):
    process = subprocess.Popen(['mpg123', '-q', path])


def listen_for_quit():
    k = cv2.waitKey(1)
    if k == ord('q'):
        return True


def get_server_response(frame, list_o_faces):
    response = post(SERVER_URL, json={'list_o_faces': list_o_faces, 'frame': frame.tolist()})  # , files=files)
    return response.json()['frame'], response.json()['people_list'], response.json()['time']


def process_frame(video_capture, face_recognizer=None):
    if not video_capture.grab():
        return
    try:
        now = datetime.now()
        ret, frame = video_capture.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            # RYAN_PLAYED = play_mp3(RYAN_SONG_PATH, RYAN_PLAYED)
            list_o_faces = []
            for x, y, w, h in faces:
                list_o_faces.append([int(x), int(y), int(w), int(h)])
            if DEV:
                # frame, people_list, time = face_recognizer.recognize_faces(frame, list_o_faces)
                frame, people_list, time = get_server_response(frame, list_o_faces)
                frame = np.array(frame)
                frame = frame.astype('uint8')
            else:
                frame, people_list, time = get_server_response(frame, list_o_faces)
                frame = np.array(frame)
                frame = frame.astype('uint8')
            # play_mp3(RYAN_SONG_PATH)
            print(people_list, datetime.now() - now)
        else:
            print(datetime.now() - now)

        if DEV:
            # Display the resulting frame
            cv2.imshow('Video', frame)
    except TypeError as error:
        print(error)
        return
    except SyntaxError as error:
        print(error)
        return
    except ValueError as error:
        print(error)
        return

face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)
if DEV:
    face_recognizer = FaceRecognizer()
    video_capture = cv2.VideoCapture(1)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

else:
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.cv.CV_CAP_PROP_FPS, 25)
    # video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    # video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
    face_recognizer = None

while True:
    process_frame(video_capture, face_recognizer)
    if listen_for_quit():
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
