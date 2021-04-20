from pydub import AudioSegment
from pathlib import Path
import numpy as np
import pandas as pd
import _pickle as pickle
import csv
from melody_utils import pitch_array_to_formatted
from sampling_utils import downsample_contour_array


class HummingDB:
    def __init__(self, data_path, audio_path, df_a, df_b):
        self.data_path = Path(data_path)
        self.audio_path = Path(audio_path)
        self.song_list = list(self.data_path.rglob('*.wav'))
        self.samples = [make_humming_sample_dictionary(path, df_a, df_b) for path in self.song_list]
        self.num_songs = len(self.song_list)
        
    def __getitem__(self, index):
        selected_sample = self.samples[index]
        contour = get_normalized_contour_from_sample(selected_sample)
        
        orig_audio_path = get_orig_audio_path_by_id(selected_sample['track_id'], self.audio_path)
        orig_pitch_path = audio_path_to_pitch_path(orig_audio_path)
        orig_contour = load_melody_txt(orig_pitch_path)
        orig_contour = pitch_array_to_formatted(orig_contour)
        # orig_contour = downsample_contour_array(orig_contour)
        # orig_contour[np.isnan(orig_contour)] = 0

        time_pos = selected_sample['time_stamp'].split('-')
        if len(time_pos) != 2:
            print(selected_sample)
        start_position = int(time_pos[0]) * 100
        end_position = int(time_pos[1]) * 100
        return {'humm':contour, 'orig':orig_contour[start_position:end_position], 'meta':selected_sample} 

    def _get_audio(self, index):
        selected_sample = self.samples[index]
        song_path = selected_sample['path']
        song = AudioSegment.from_file(song_path, 'wav')._data
        decoded = np.frombuffer(song, dtype=np.int16) / 32768
        
        orig_audio_path = get_orig_audio_path_by_id(selected_sample['track_id'], self.audio_path)
        orig_song = AudioSegment.from_file(orig_audio_path, 'm4a').set_channels(1)._data
        orig_decoded = np.frombuffer(orig_song, dtype=np.int16) / 32768
        
        time_pos = selected_sample['time_stamp'].split('-')
        start_position = int(time_pos[0]) * 44100
        end_position = int(time_pos[1]) * 44100
                        
        return decoded, orig_decoded[start_position:end_position], selected_sample

    def __len__(self):
        return len(self.samples)

def get_normalized_contour_from_sample(selected_sample):
    humm_melody = load_crepe_pitch(selected_sample['pitch_path'])
    humm_melody = pitch_array_to_formatted(humm_melody)
    # humm_melody = downsample_contour_array(humm_melody)
    # humm_melody[np.isnan(humm_melody)] = 0
    return humm_melody

def get_orig_audio_path_by_id(track_id, audio_dir):
    track_id = str(track_id)
    orig_audio_path = audio_dir / track_id[:3] / track_id[3:6] / (track_id +'.aac')
    if not orig_audio_path.exists():
        orig_audio_path = orig_audio_path.with_suffix('.m4a')
    if not orig_audio_path.exists():
        orig_audio_path = audio_dir / 'qbh' / (track_id + '.aac')
    return orig_audio_path

def audio_path_to_pitch_path(path):
    return path.parent / f'{path.stem}_pitch.txt'

def load_melody_txt(path, to_midi_pitch=True):
    with open(path, "r") as f:
        lines = f.readlines()
    data = np.asarray([float(x.split(' ')[1][:-2]) for x in lines])
    if to_midi_pitch:
        data[data>0] = np.log2(data[data>0]/440) * 12 + 69
    return data

def make_humming_sample_dictionary(path, df_a, df_b):
    sample = {}
    meta = path.stem.split('_')
    sample['path'] = str(path)
    sample['pitch_path'] = str(path.with_suffix('.f0.csv'))

    if meta[0] == "100":
        sample['song_group'], sample['song_idx'], sample['humming_type'], sample['time_stamp'], sample['singer_group'], sample['singer_id'] = meta
        sample['singer_gender'] = sample['singer_group'][2]
        sample['singer_group'] = sample['singer_group'][1]
        row = df_a.loc[df_a['file_name'] == path.name].iloc[0]
        sample['track_id'] = row['track_id']
        sample['singer_id'] = sample['singer_id'][:-1]
        
    else:
        sample['song_group'], sample['song_idx'], sample['humming_type'], sample['time_stamp'] = meta
        
        row = df_b.loc[df_b['file_name'] == path.name].iloc[0]
        sample['track_id'] = row['track_id']
        sample['singer_gender'] = row['Identification code'][1]
        sample['singer_group'] = row['Identification code'][0]
        sample['singer_id'] = row['Identification code'][-3:]
    
    return sample
    

def load_meta_from_excel(xlsx_path="/home/svcapp/userdata/humming_db/Spec.xlsx", meta_path="flo_metadata.dat", meta_100_path='meta_100.dat'):
    xls_file = pd.ExcelFile(xlsx_path)
    sheets = pd.read_excel(xls_file, sheet_name=None, header=1)
    exp_id = list(sheets.keys())
    selected_100 = [sheets[x] for x in exp_id[:4]]
    selected_100 = pd.concat(selected_100, ignore_index=True)
    selected_900 = sheets[exp_id[4]]

    with open(meta_path, "rb") as f:
        data_dict = pickle.load(f)
    with open(meta_100_path, 'rb') as f:
        meta_100 = pickle.load(f)

    track_ids = [get_track_id(selected_900['track_name'][x], selected_900['artist_name'][x], data_dict) for x in range(900) ]
    track_ids100 =  [get_track_id(selected_100['track_name'][x], selected_100['artist_name'][x], meta_100) for x in range(500) ]

    selected_100['track_id'] = track_ids100
    selected_900['track_id'] = track_ids
    return selected_100, selected_900

def get_track_id(song_name, artist_name, data_dict):
    for song in data_dict:
        if song_name == str(song['track_name']) and str(artist_name) in str(song['artist_name_basket'][0]):
            return song['track_id']
    print(f"{song_name} / {artist_name}")

def load_pitch_csv(pitch_path):
    with open(pitch_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.asarray(data[1:], dtype='float32')
    return data

def load_crepe_pitch(pitch_path, threshold=0.7, to_midi_pitch=True, cut_low_frequency=True):
    pitch_data = load_pitch_csv(pitch_path)
    pitch_data[pitch_data[:,2]<threshold, 1] = 0
    pitch_data = pitch_data[:,1]
    if cut_low_frequency:
        pitch_data[pitch_data<80] = 0
    if to_midi_pitch:
        pitch_data[pitch_data>0] = np.log2(pitch_data[pitch_data>0]/440) * 12 + 69

    return pitch_data


if __name__ == "__main__":
    selected_100, selected_900 = load_meta_from_excel()
    humming_db = HummingDB('/home/svcapp/userdata/humming_db', '/home/svcapp/userdata/flo_data_backup/', selected_100, selected_900)
    # contour_pairs = [humming_db[i] for i in range(len(humming_db))]
    contour_pairs = [x for x in humming_db]
    with open('humming_db_contour_pairs.dat', "wb") as f:
        pickle.dump(contour_pairs, f)
