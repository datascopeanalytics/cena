import os
from datetime import datetime
from glob import glob

import cv2
import dlib
import numpy as np
import openface
import pandas as pd
from sklearn.svm import SVC

from cena.settings import (SHAPE_PREDICTOR_FILE_PATH, CASCADE_FILE_PATH, FEATURE_EXTRACTOR_FILE_PATH,
                           DEV, LABELS_FILE_PATH, REPS_FILE_PATH, ANNOTATE_FRAME)


class FaceRecognizer(object):
    def __init__(self):
        self.face_pose_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_FILE_PATH)
        print('loaded face pose predictor')
        self.face_aligner = openface.AlignDlib(SHAPE_PREDICTOR_FILE_PATH)
        print('loaded face aligner')

        self.face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)
        print('loaded face cascade')
        self.clf, self.user_list = self.train_model()
        self.net = openface.TorchNeuralNet(FEATURE_EXTRACTOR_FILE_PATH)
        print('loaded torch nn')

    def train_model(self):
        labels = pd.read_csv(LABELS_FILE_PATH, header=None).rename(columns={0: 'label', 1: 'user'})
        labels.user = labels.user.apply(lambda x: x.split('/')[-2])
        user_list = labels.user.unique()
        x = pd.read_csv(REPS_FILE_PATH, header=None)

        clf = SVC(C=1, kernel='linear', probability=True)
        clf.fit(x, labels.user)
        print('classifier trained')
        print('users found:', user_list)
        return clf, user_list

    def output_training_features(self, inpath, outpath):
        frame = cv2.imread(inpath)
        height, width = frame.shape[:2]
        rect = dlib.rectangle(left=0, top=0, right=width, bottom=height)
        # pose_landmarks = self.face_pose_predictor(frame, rect)

        alignedFace = self.face_aligner.align(534, frame, rect,
                                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        cv2.imwrite(outpath, alignedFace)

    def make_training_set(self, directory='data/img/*', out_dir='data/transformed_img/'):
        files = glob(directory)
        for file_path in files:
            self.output_training_features(file_path, os.path.join(out_dir + file_path.split('/')[-1]))

    def recognize_faces(self, frame, list_o_faces):
        start = datetime.now()
        pred_names = []
        for x, y, w, h in list_o_faces:
            rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
            aligned_face = self.face_aligner.align(96, frame, rect,
                                                   landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            nn_features = self.net.forward(aligned_face)
            # pred_name = self.clf.predict([nn_features])[0]
            pred_probs = self.clf.predict_proba([nn_features])[0]
            highest_prob_index = np.argmax(pred_probs)
            pred_name = self.clf.classes_[highest_prob_index]
            pred_prob = max(pred_probs)
            pred_names.append({pred_name: pred_prob})

            if ANNOTATE_FRAME:
                pose_landmarks = self.face_pose_predictor(frame, rect)
                cv2.putText(frame, '{}: {}'.format(pred_name, pred_prob), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (102, 204, 102), thickness=2)
                for point in pose_landmarks.parts():
                    x, y = point.x, point.y
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            end = datetime.now()
            return frame, pred_names, (end - start).microseconds / 1000

            # if DEV and ANNOTATE_FRAME:
            #     pose_landmarks = self.face_pose_predictor(frame, rect)
            #     cv2.putText(frame, '{}: {}'.format(pred_name, pred_prob), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                 (102, 204, 102), thickness=2)
            #     for point in pose_landmarks.parts():
            #         x, y = point.x, point.y
            #         cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            #
            #     end = datetime.now()
            #     return frame, pred_names, (end - start).microseconds / 1000
            # else:
            #     end = datetime.now()
            #     return pred_names, (end - start).microseconds / 1000

