import glob
import cv2
import numpy as np


path = 'dataSet/*'
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

imgs = []
labels = []

for f in glob.glob(path):
    i = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    i_resize = cv2.resize(i,(224,224))
    imgs.append(i_resize)
    labels.append(int(f.split('.')[1]))

recognizer.train(imgs, np.array(labels))
recognizer.write('trained_recognizer.yaml')