import numpy as np
import base64


def encode_image(image):
    image = image.astype(np.uint8)
    return base64.b64encode(image.tobytes()).decode('utf-8')


def decode_image(encoded_str, shape):
    decoded_arr = np.fromstring(base64.b64decode(encoded_str), dtype=np.uint8)
    return decoded_arr.reshape(shape)
