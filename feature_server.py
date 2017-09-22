from cena.recognition import FaceRecognizer


class FeatureServer(object):
    def __init__(self):
        self.recognizer = FaceRecognizer()
