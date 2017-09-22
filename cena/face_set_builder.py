import cv2
import os

MODELS_DIR = '../data/models/'
DATA_DIR = '../data/img/'
CASCADE_FILE_NAME = 'haarcascade_frontalface_default.xml'
CASCADE_FILE_PATH = os.path.join(MODELS_DIR, CASCADE_FILE_NAME)

cam = cv2.VideoCapture(2)
detector = cv2.CascadeClassifier(CASCADE_FILE_PATH)

Id = input('enter your id')
sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # incrementing sample number
        sampleNum = sampleNum + 1
        # saving the captured face in the dataset folder
        cv2.imwrite(DATA_DIR + "User." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', img)
    # wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum > 30:
        break
cam.release()
cv2.destroyAllWindows()