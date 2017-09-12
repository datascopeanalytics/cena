import cv2
import sys
import numpy as np

names = {2: 'Nat', 3: 'Jess', 4: 'Mike', 5: 'Irmak', 6: 'Vlad',
         7: 'Chris', 8: 'Francesca'}
poo = cv2.imread('poo.png')

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_recognizer.yaml')

video_capture = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
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
        label, conf = recognizer.predict(face_resize)

        if conf < 65:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (171, 21, 255), 2)

            cv2.putText(frame, '{}: {:.0f}'.format(names[label], conf),
                        (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 204, 102),
                        thickness=2)
            if label == 4:
                poo_head = cv2.resize(poo, (w, h),
                                      interpolation=cv2.INTER_CUBIC)
                frame[y: y + h, x: x + w] = poo_head
                cv2.putText(frame, 'DOO-DOO HEAD ALERT!!',
                            (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (171, 21, 255),
                            thickness=2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


