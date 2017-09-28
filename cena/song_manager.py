from glob import glob

from cena.settings import SONGS_DIR


def get_name_from_path(file_path):
    file_name = file_path.split('/')[-1]
    person_name = file_name.split('.')[0]
    return person_name


class SongManager(object):
    def __init__(self):
        self.song_files = song_files =glob(SONGS_DIR + '*.*')
        self.person_songs = {get_name_from_path(file_name): file_name
                             for file_name in song_files}

