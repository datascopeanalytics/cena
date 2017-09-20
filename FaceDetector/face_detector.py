import cv2
import dlib
import openface
import os
import numpy as np
from datetime import datetime
from glob import glob
import pandas as pd
from sklearn.svm import SVC
import subprocess

MODELS_DIR = '../data/models/'
SONGS_DIR = '../data/songs/'

CASCADE_FILE_NAME = 'haarcascade_frontalface_default.xml'
SHAPE_PREDICTOR_FILE_NAME = 'shape_predictor_68_face_landmarks.dat'
FEATURE_EXTRACTOR_FILE_NAME = 'nn4.small2.v1.t7'

CASCADE_FILE_PATH = os.path.join(MODELS_DIR, CASCADE_FILE_NAME)
SHAPE_PREDICTOR_FILE_PATH = os.path.join(MODELS_DIR, SHAPE_PREDICTOR_FILE_NAME)
FEATURE_EXTRACTOR_FILE_PATH = os.path.join(MODELS_DIR, FEATURE_EXTRACTOR_FILE_NAME)

RYAN_FILE_NAME = 'dun_dun_dun.mp3'
RYAN_SONG_PATH = os.path.join(SONGS_DIR, RYAN_FILE_NAME)

ENVIRONMENT = 'dev'
# ENVIRONMENT = 'nah dude'


def play_mp3(path):
    process = subprocess.Popen(['mpg123', '-q', path])


class FaceRecognizer(object):
    def __init__(self):
        self.face_pose_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_FILE_PATH)
        print('loaded face pose predictor')
        self.face_aligner = openface.AlignDlib(SHAPE_PREDICTOR_FILE_PATH)
        print('loaded face aligner')

        self.face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)
        print('loaded face cascade')
        self.clf = self.train_model()
        self.net = openface.TorchNeuralNet(model=FEATURE_EXTRACTOR_FILE_PATH)
        print('loaded torch nn')

    def train_model(self, train_data_path='../data/generated-embeddings/'):
        labels = pd.read_csv(os.path.join(train_data_path, 'labels.csv'), header=None).rename(columns={0:'label', 1:'user'})
        labels.user = labels.user.apply(lambda x: x.split('/')[-2])
        x = pd.read_csv(os.path.join(train_data_path, 'reps.csv'), header=None)

        clf = SVC(C=1, kernel='linear', probability=True)
        clf.fit(x, labels.user)
        print('classifier trained')
        return clf

    def output_training_features(self, inpath, outpath):
        frame = cv2.imread(inpath)
        height, width = frame.shape[:2]
        rect = dlib.rectangle(left=0, top=0, right=width, bottom=height)
        # pose_landmarks = self.face_pose_predictor(frame, rect)

        alignedFace = self.face_aligner.align(534, frame, rect,
                                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        cv2.imwrite(outpath, alignedFace)

    def make_training_set(self, directory='../data/img/*', out_dir='../data/transformed_img/'):
        files = glob(directory)
        for file_path in files:
            self.output_training_features(file_path, os.path.join(out_dir + file_path.split('/')[-1]))

    def recognize_faces(self, frame, list_o_faces):
        start = datetime.now()
        pred_names = []
        for x, y, w, h in list_o_faces:
            rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
            pose_landmarks = self.face_pose_predictor(frame, rect)

            aligned_face = self.face_aligner.align(96, frame, rect,
                                                   landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            nn_features = self.net.forward(aligned_face)
            # pred_name = self.clf.predict([nn_features])[0]
            pred_probs = self.clf.predict_proba([nn_features])[0]
            highest_prob_index = np.argmax(pred_probs)
            pred_name = self.clf.classes_[highest_prob_index]
            pred_prob = max(pred_probs)
            pred_names.append({pred_name:pred_prob})

            cv2.putText(frame, '{}: {}'.format(pred_name, pred_prob), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (102, 204, 102), thickness=2)
            for point in pose_landmarks.parts():
                x, y = point.x, point.y
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        end = datetime.now()
        return frame, pred_names, (end - start).microseconds / 1000

face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)
face_recognizer = FaceRecognizer()

if ENVIRONMENT == 'dev':
    video_capture = cv2.VideoCapture(1)
else:
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.cv.CV_CAP_PROP_FPS, 25)
    # video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    # video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

while True:
    if not video_capture.grab():
        continue
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

        images = []
        labels = []

        # Draw a rectangle around the faces
        if len(faces) > 0:
            # RYAN_PLAYED = play_mp3(RYAN_SONG_PATH, RYAN_PLAYED)
            list_o_faces = []
            for x, y, w, h in faces:
                list_o_faces.append([int(x), int(y), int(w), int(h)])
            frame, person, time = face_recognizer.recognize_faces(frame, list_o_faces)
            print(person, time)
        else:
            print(datetime.now() - now)

        if ENVIRONMENT == 'dev':
            # Display the resulting frame
            cv2.imshow('Video', frame)
    except TypeError as error:
        print(error)
        continue
    except SyntaxError as error:
        print(error)
        continue
    except ValueError as error:
        print(error)
        continue

    if ENVIRONMENT == 'dev':
        k = cv2.waitKey(1)
        if k == ord('q'):
            break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
