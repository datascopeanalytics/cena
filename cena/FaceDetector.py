# Face detection on the jevois
# We're not using this but keeping it for posterity

import libjevois as jevois
import cv2
import os


class FaceDetector:
    # ###################################################################################################
    # Constructor
    def __init__(self):
        code_path = '/jevois/modules/JeVois/cena'

        cascPath = os.path.join(code_path, 'haarcascade_frontalface_default.xml')
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    # ###################################################################################################
    # Process function with USB output
    def process(self, inframe, outframe):
        frame = inframe.getCvGRAY()
        # frame = inframe.getCvBGR()
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # if there are faces, send them in a list o lists via serial
        if len(faces) > 0:
            face_list = []
            for x, y, w, h in faces:
                face_list.append([x, y, w, h])
            jevois.sendSerial(str(face_list))

        # always send the output frame
        outframe.sendCvBGR(frame)
