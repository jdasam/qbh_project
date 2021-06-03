from pathlib import Path
from test_qbh import QbhSystem
import argparse
from utils.data_utils import get_song_ids_of_selected_genre, song_id_to_pitch_txt_path
import _pickle as pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        default="emb_test",
                        help='directory to save embedding')
    parser.add_argument('-c', '--checkpoint_directory', type=Path, 
                        default=Path("weights/contour_model"),
                        help='directory to load checkpoint')
    parser.add_argument('--dataset_dir', type=Path, 
                        default= Path('/home/svcapp/t2meta/flo_new_music/music_100k/'),
                        help='directory to load checkpoint') 
    parser.add_argument('--meta_dat_path', type=str, 
                    default= 'data/flo_metadata_220k.dat',
                    help='directory to load checkpoint')
    parser.add_argument('--humm_db_ids_path', type=str, 
                default= 'data/humm_db_ids.dat',
                help='directory to load checkpoint')       
    parser.add_argument('--device', type=str, default='cuda',
                             help='cpu or cuda')
    args = parser.parse_args()

    with open(args.meta_dat_path, 'rb') as f:
        db_metadata = pickle.load(f)
    # selected_genres = [4, 12, 13, 17, 10, 7,15, 11, 9]
    selected_genres= [4]
    song_ids = get_song_ids_of_selected_genre(db_metadata, selected_genres)

    with open(args.humm_db_ids_path, 'rb') as f:
        humm_ids = pickle.load(f)
        # list of humming ids
    song_ids += humm_ids

    qbh_system = QbhSystem(args.checkpoint_directory, args.output_directory, 'cuda', audio_dir=args.dataset_dir, make_emb=True, song_ids=song_ids)