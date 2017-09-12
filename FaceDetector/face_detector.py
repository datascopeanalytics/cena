import cv2
import dlib
import openface
import sys
import os
import numpy as np
from ast import literal_eval
from time import sleep
from glob import glob
import pandas as pd
from sklearn.svm import SVC

CASCADE_PATH = 'haarcascade_frontalface_default.xml'


class FaceRecognizer(object):
    recognizer_algo = None

    def __init__(self):
        self.predictor_model_path = "shape_predictor_68_face_landmarks.dat"
        self.face_pose_predictor = dlib.shape_predictor(self.predictor_model_path)
        self.face_aligner = openface.AlignDlib(self.predictor_model_path)

        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.clf = self.train_model()

    def train_model(self, train_data_path='../data/generated-embeddings/'):
        labels = pd.read_csv(os.path.join(train_data_path, 'labels.csv'), header=None).rename(columns={0:'label', 1:'user'})
        labels.user = labels.user.apply(lambda x: x.split('/')[-2])
        x = pd.read_csv(os.path.join(train_data_path, 'reps.csv'), header=None)

        clf = SVC()
        clf.fit(x, labels.user)
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

    def recognize_faces(self, frame, face_list_str):
        list_o_faces = literal_eval(face_list_str)

        for x, y, w, h in list_o_faces:
            cropped_face = frame[y:y+h, x:x+w]
            rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
            pose_landmarks = self.face_pose_predictor(frame, rect)

            alignedFace = self.face_aligner.align(534, frame, rect,
                                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            # if len(pose_landmarks.parts()) > 0:
            for point in pose_landmarks.parts():
                x, y = point.x, point.y
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # print(pose_landmarks)
            if self.recognizer_algo is not None:
                found_face = self.recognizer_algo.better_recognize(cropped_face)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (171, 21, 255), 2)
        return frame
#
# face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
# video_capture = cv2.VideoCapture(2)
# face_recognizer = FaceRecognizer()
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # frame = inframe.getCvBGR()
#     # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(
#         frame,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
#
#     # if there are faces, send them in a list o lists via serial
#     if len(faces) > 0:
#         face_list = []
#         for x, y, w, h in faces:
#             face_list.append([x, y, w, h])
#         face_list_str = str(face_list)
#
#         frame = face_recognizer.recognize_faces(frame, face_list_str)
#
#     # Display the resulting frame
#     cv2.imshow('Video', frame)
#
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break
#
# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()