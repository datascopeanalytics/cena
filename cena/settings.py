import os
import pytz
from ast import literal_eval

ENVIRONMENT = os.getenv('FACE_ENV', 'lol')
DEV = ENVIRONMENT == 'dev'

# YOLO_MODE = True
YOLO_MODE = False

# ANNOTATE_FRAME = True
ANNOTATE_FRAME = False

CLIENT_ENV_VAR = os.getenv('FACE_CLIENT', True)
if not isinstance(CLIENT_ENV_VAR, bool):
    IS_CLIENT = literal_eval(CLIENT_ENV_VAR)
else:
    IS_CLIENT = CLIENT_ENV_VAR

API_SERVER_NAME = 'face-api'

MODELS_DIR = 'data/models/'
SONGS_DIR = 'data/songs/'

CASCADE_FILE_NAME = 'haarcascade_frontalface_default.xml'
SHAPE_PREDICTOR_FILE_NAME = 'shape_predictor_68_face_landmarks.dat'
if DEV:
    FEATURE_EXTRACTOR_FILE_NAME = 'nn4.small2.v1.t7'
else:
    FEATURE_EXTRACTOR_FILE_NAME = 'nn4.small2.v1.t7'

CASCADE_FILE_PATH = os.path.join(MODELS_DIR, CASCADE_FILE_NAME)
SHAPE_PREDICTOR_FILE_PATH = os.path.join(MODELS_DIR, SHAPE_PREDICTOR_FILE_NAME)
FEATURE_EXTRACTOR_FILE_PATH = os.path.join(MODELS_DIR, FEATURE_EXTRACTOR_FILE_NAME)

TRAIN_DATA_PATH = 'data/generated-embeddings/'
LABELS_FILE_NAME = 'labels.csv'
REPS_FILE_NAME = 'reps.csv'

LABELS_FILE_PATH = os.path.join(TRAIN_DATA_PATH, LABELS_FILE_NAME)
REPS_FILE_PATH = os.path.join(TRAIN_DATA_PATH, REPS_FILE_NAME)


# Todo: get these user/song file paths via file names
RYAN_FILE_NAME = 'dun_dun_dun.mp3'
RYAN_SONG_PATH = os.path.join(SONGS_DIR, RYAN_FILE_NAME)

if not DEV:
    from cena.utils import get_api_server_ip_address
    SERVER_IP = get_api_server_ip_address()
else:
    SERVER_IP = 'localhost'

SERVER_URL = 'http://{}:5000/recognize'.format(SERVER_IP)

TIME_ZONE = pytz.timezone('America/Chicago')

WINDOW_SIZE = 10
MIN_SEEN = 3
PROBA_THRESHOLD = 0.4
