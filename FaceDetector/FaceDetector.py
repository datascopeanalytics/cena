import libjevois as jevois
import cv2
import numpy as np
import os 

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module is here for you to experiment with Python OpenCV on JeVois.
#
# By default, we get the next video frame from the camera as an OpenCV BGR (color) image named 'inimg'.
# We then apply some image processing to it to create an output BGR image named 'outimg'.
# We finally add some text drawings to outimg and send it to host over USB.
#
# @author Laurent Itti
# 
# @videomapping YUYV 352 288 30.0 YUYV 352 288 30.0 JeVois PythonSandbox
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules

class FaceDetector:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        self.names = {2: 'Nat', 3: 'Jess', 4: 'Mike', 5: 'Irmak', 6: 'Vlad',
                      7: 'Chris', 8: 'Francesca'}
        self.poo = cv2.imread('poo.png')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(dir_path)

        cascPath = 'haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(cascPath)

        # recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer = cv2.face.createLBPHFaceRecognizer()
        file_path = '/jevois/modules/JeVois/PythonSandbox/trained_recognizer.yaml'
        recognizer.load(file_path)
        # try:
        #     with open(file_path, 'rb') as f:
        #         recognizer.load(f)
        # except FileNotFoundError as x:
        #     raise FileNotFoundError(help(recognizer))
        #     print(dir_path)
        #     print(x)
        #     raise dir_path

        self.recognizer = recognizer
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        frame = inframe.getCvBGR()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        images = []
        labels = []

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            face_resize = cv2.resize(frame_gray[y: y + h, x: x + w], (224, 224))
            label, conf = self.recognizer.predict(face_resize)

            if conf < 65:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (171, 21, 255), 2)

                cv2.putText(frame, '{}: {:.0f}'.format(names[label], conf),
                            (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 204, 102),
                            thickness=2)
                if label == 4:
                    poo_head = cv2.resize(self.poo, (w, h),
                                          interpolation=cv2.INTER_CUBIC)
                    frame[y: y + h, x: x + w] = poo_head
                    cv2.putText(frame, 'DOO-DOO HEAD ALERT!!',
                                (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (171, 21, 255),
                                thickness=2)

        outframe.sendCvBGR(frame)
