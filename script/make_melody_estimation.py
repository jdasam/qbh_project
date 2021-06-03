from pathlib import Path
from test_qbh import QbhSystem
import argparse
from utils.data_utils import get_song_ids_of_selected_genre
from utils.melody_utils import MelodyLoader
import _pickle as pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, 
                        default= Path('/home/svcapp/t2meta/flo_new_music/music_100k/'),
                        help='directory to load checkpoint')     
    args = parser.parse_args()

    with open("data/flo_metadata_220k.dat", 'rb') as f:
        db_metadata = pickle.load(f)
    selected_genres = [4, 12, 13, 17, 10, 7,15, 11, 9]
    song_ids = get_song_ids_of_selected_genre(db_metadata, selected_genres)

    with open('data/humm_db_ids.dat', 'rb') as f:
        humm_ids = pickle.load(f)
    song_ids += humm_ids


    melody_loader = MelodyLoader(args.dataset_dir)
    melody_loader.check_txt_exists(song_ids, make_txt=True, verbose=True) 