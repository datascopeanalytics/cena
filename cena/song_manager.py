from subprocess import call
from glob import glob

from cena.settings import SONGS_DIR, WINDOW_SIZE, PROBA_THRESHOLD, YOLO_MODE, MIN_SEEN
from cena.utils import play_mp3


def get_name_from_path(file_path):
    file_name = file_path.split('/')[-1]
    person_name = file_name.split('.')[0]
    return person_name


class SongManager(object):
    def __init__(self):
        self.is_blank_slate = True

        self.song_files = song_files = glob(SONGS_DIR + '*.*')
        self.person_songs = {get_name_from_path(file_name): file_name
                             for file_name in song_files}
        self.people = self.person_songs.keys()

        self.person_thresholds = {person: 0. for person in self.people}
        self.played_today = self.make_new_slate()
        self.window = []

    def make_new_slate(self):
        return {person: 0 for person in self.people}

    def _person_found(self, person):
        people_mask = [p == person for p in self.window]
        total_found = sum(people_mask)

        more_than_half = total_found >= int(WINDOW_SIZE / 2)
        half_of_seen = total_found >= int(len(self.window) / 2)
        more_than_min = total_found > MIN_SEEN
        return more_than_half and more_than_min and half_of_seen

    def update_window(self, person, proba):
        if proba > PROBA_THRESHOLD:
            self.window.append(person)
        if len(self.window) > WINDOW_SIZE:
            self.window.pop(0)

        if self._person_found(person):
            # print(self.window)
            try:
                self.go_song_go(person)
            except KeyError as error:
                print('oh whoops no song for {}'.format(person))

    def go_song_go(self, person):
        if self.played_today[person] < 1:
            print('playing that funky music for {}'.format(person))
            play_mp3(self.person_songs[person])
            self.window = []

            if not YOLO_MODE:
                self.played_today[person] = 1
        else:
            # print('you\'ve already had your fill today {}'.format(person))
            pass

    def blank_the_slate(self):
        if self.is_blank_slate:
            return
        self.played_today = self.make_new_slate()
        self.window = []
        print('oh wow such reset')

    # may not be the right place, but don't want to forget
    def update_dropbox(self):
        # fixme: make this the right command
        command = "/home/pi/Dropbox-Uploader/dropbox_uploader.sh download /songs ~/cena/data/songs"
        call([command], shell=True)