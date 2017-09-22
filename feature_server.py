from flask import Flask, request, jsonify, abort
import numpy as np
import cv2
from scipy.misc import toimage

from cena.recognition import FaceRecognizer

RECOGNIZER = FaceRecognizer()
app = Flask(__name__)


@app.route('/recognize', methods=['POST'])
def recognize():
    if not request.json:
        abort(400)
    if 'frame' not in request.json:
        abort(400)
    if 'list_o_faces' not in request.json:
        abort(400)
    # frame = cv2.imdecode(np.array(request.json['frame']), None)
    frame = np.array(request.json['frame'])
    frame = frame.astype('uint8')
    # frame = toimage(np.array(request.json['frame']))
    # frame = cv2.(np.array(request.json['frame']))
    # print(frame[0][0].dtype)
    list_o_faces = request.json['list_o_faces']

    # print(type(list_o_faces[0][0]))

    frame, people_list, time = RECOGNIZER.recognize_faces(frame, list_o_faces)
    response = {
        'people_list': people_list,
        'frame': frame.tolist(),
        'time': time
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
