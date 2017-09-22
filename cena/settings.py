import os

ENVIRONMENT = 'dev'
# ENVIRONMENT = 'nah dude'

DEV = ENVIRONMENT == 'dev'

MODELS_DIR = 'data/models/'
SONGS_DIR = 'data/songs/'


CASCADE_FILE_NAME = 'haarcascade_frontalface_default.xml'
SHAPE_PREDICTOR_FILE_NAME = 'shape_predictor_68_face_landmarks.dat'
if ENVIRONMENT == 'dev':
    FEATURE_EXTRACTOR_FILE_NAME = 'nn4.small2.v1.t7'
else:
    FEATURE_EXTRACTOR_FILE_NAME = 'nn4.small2.v1.ascii.t7'

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