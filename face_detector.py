import cv2
from requests import post
from datetime import datetime

from cena.recognition import FaceRecognizer
from cena.settings import DEV, ANNOTATE_FRAME, CASCADE_FILE_PATH, SERVER_URL, TIME_ZONE
from cena.song_manager import SongManager
from cena.utils import encode_image, decode_image, play_mp3


def listen_for_quit():
    k = cv2.waitKey(1)
    if k == ord('q'):
        return True


def get_server_response(frame, list_o_faces, return_frame=False):
    # response = post(SERVER_URL, json={'list_o_faces': list_o_faces, 'frame': frame.tolist()})  # , files=files)
    shape = frame.shape
    request_json = {
        'list_o_faces': list_o_faces,
        'frame': encode_image(frame),
        'shape': shape,
        'return_frame': return_frame
    }
    response = post(SERVER_URL, json=request_json)

    people_list = response.json()['people_list']
    time = response.json()['time']

    if return_frame:
        frame = decode_image(response.json()['frame'], shape)
    return frame, people_list, time
    # return response.json()['frame'], response.json()['people_list'], response.json()['time']


def process_frame(video_capture, face_recognizer=None):
    if not video_capture.grab():
        return
    try:
        now = datetime.now(TIME_ZONE)
        if now.hour > 21:
            song_manager.blank_the_slate()

        ret, frame = video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # ret, frame = video_capture.read(cv2.COLOR_BGR2GRAY)
        # ret, frame = video_capture.read(cv2.IMREAD_GRAYSCALE)
        frame_gray = frame
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = frame_gray
        faces = face_cascade.detectMultiScale(
            frame_gray,
            # scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            # minSize=(60, 60),
            # flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            list_o_faces = []
            for x, y, w, h in faces:
                list_o_faces.append([int(x), int(y), int(w), int(h)])
            if DEV:
                frame, people_list, time = face_recognizer.recognize_faces(frame, list_o_faces)
                # frame, people_list, time = get_server_response(frame, list_o_faces, ANNOTATE_FRAME)
            else:
                frame, people_list, time = get_server_response(frame, list_o_faces, ANNOTATE_FRAME)
            for person, proba in people_list.items():
                song_played = song_manager.update_window(person, proba)
                if song_played:
                    print(people_list, datetime.now(TIME_ZONE) - now)
        else:
            pass
            # print(datetime.now(TIME_ZONE) - now)

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

song_manager = SongManager()
person_songs = song_manager.person_songs
print('found songs:')
print(person_songs)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)
if DEV:
    face_recognizer = FaceRecognizer()
    trained_people = face_recognizer.user_list
    print('people i recognize but do not have a song for:')
    print([i for i in trained_people if i not in person_songs])

    video_capture = cv2.VideoCapture(1)
    video_capture.set(cv2.CAP_PROP_FPS, 5)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

else:
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.cv.CV_CAP_PROP_FPS, 5)
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
