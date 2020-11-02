import numpy as np
import copy
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
from melody_utils import binary_index
from time import time
from sampling_utils import downsample_contour

def normalize_contour(windowed_contour):
    pitch_mean = np.sum(windowed_contour) / np.count_nonzero(windowed_contour)
    nonzero_indices = np.nonzero(windowed_contour)
    zero_indices=np.where(windowed_contour == 0)[0]
    windowed_contour[nonzero_indices] -= pitch_mean
    windowed_contour[zero_indices] = -100
    
    return windowed_contour

def make_pitch_vector(contour_dict, melody_idx, down_f=10, window_size=100, hop_size=10, min_mel_ratio=0.6):
    contour = contour_dict['melody']
    ds_contour = downsample_contour(contour, down_f=down_f)
    num_pitch_vector = ds_contour.shape[0] // hop_size
    c_idx = 0
    outputs = []
    for i in range(num_pitch_vector):
        if np.count_nonzero(ds_contour[c_idx:c_idx+window_size]) / window_size > min_mel_ratio:
            norm_contour = normalize_contour(np.copy(ds_contour[c_idx:c_idx+window_size]))
            if norm_contour.shape[0] < window_size:
                dummy = np.zeros(window_size, dtype=float)
                dummy[:norm_contour.shape[0]] = norm_contour
                norm_contour = dummy
            temp_dict = {'song_id': contour_dict['song_id'],
                         'frame_pos': contour_dict['frame_pos'],
                         'detailed_frame_pos': (contour_dict['frame_pos'][0] + down_f*c_idx, contour_dict['frame_pos'][0] + down_f*(c_idx+window_size)),
                         'pitch_vector': norm_contour,
                         'melody_idx': melody_idx
                        }
            outputs.append(temp_dict)
        c_idx += hop_size
    return outputs


def remove_zero_pitch(contour):
    new_contour = np.copy(contour)
    first_non_zero_pitch = new_contour[np.where(new_contour!=-100)[0][0]]
    if new_contour[0] == -100:
        new_contour[0] = first_non_zero_pitch
    for i in range(1, len(new_contour)):
        if new_contour[i] == -100:
            new_contour[i] = new_contour[i-1]
    return new_contour

def remove_zero_pitch_in_array(contour):
    new_contour = np.copy(contour)
    for i in range(contour.shape[0]):
        new_contour[i,:] = remove_zero_pitch(contour[i,:])
    return new_contour

def compare_pitch_vector_without_zero(query, target):
    query = remove_zero_pitch(query)
    zero_indices = np.where(target==-100)[0]
    assert query.shape == target.shape
    distance = np.sqrt((query-target) ** 2)
    distance[zero_indices] = 0
    print(distance)
    return np.sum(distance) / (target.shape[0] - zero_indices.shape[0])

class PitchVecDB:
    def __init__(self, contour_list, audio_dir, down_f=10, window_size=100, hop_size=10, min_overlap_ratio=0.8, sr=44100):
        self.contours = contour_list
        pitch_vectors = [make_pitch_vector(x, down_f=down_f, window_size=window_size, hop_size=hop_size, melody_idx=i) for i, x in enumerate(tqdm(contour_list))]
        self.pitch_vectors = [y for x in pitch_vectors for y in x]
        db_array = [x['pitch_vector'] for x in self.pitch_vectors]
        self.db = np.asarray(db_array)
        self.is_nonzero = self.db !=-100
        self.zero_idc = np.where(self.db==-100)
        self.num_pitch_by_vector = np.count_nonzero(self.db!=-100, axis=1)
        self.down_f = down_f
        self.window_size = window_size
        self.hop_size = hop_size
        self.min_overlap_ratio = min_overlap_ratio
        self.non_zero_db = np.asarray([remove_zero_pitch(x) for x in self.db])
        self.audio_dir = Path(audio_dir)
        self.sr= sr
        self.is_nonzero_sum = np.sum(self.is_nonzero, axis=1)
        nan_db = np.copy(self.db)
        nan_db[nan_db==-100] = np.nan
        self.db_std = np.nanstd(nan_db, axis=1)
    
    def search_query_from_db(self, query, topk=5):
        nonzero_idc_query= query!=-100
        overlap = np.sum(nonzero_idc_query * self.is_nonzero, axis=1)
        nan_query = np.copy(query)
        nan_query[nan_query==-100] = np.nan
        query_std = np.nanstd(nan_query)
        query = remove_zero_pitch(query)
        condition_a = overlap > np.sum(nonzero_idc_query) * self.min_overlap_ratio
        condition_b = overlap > self.is_nonzero_sum * self.min_overlap_ratio
        condition_c = np.abs(self.db_std - query_std) < 0.15
        # valid_overlap_idx = np.where( (overlap> np.sum(nonzero_idc_query) * self.min_overlap_ratio) * (overlap>self.is_nonzero_sum*self.min_overlap_ratio) )[0]
        valid_overlap_idx = np.where(condition_a * condition_b * condition_c)[0]
        if valid_overlap_idx.shape[0] < 3:
            return [], [], []

        valid_db = self.db[valid_overlap_idx]
        diff = np.sqrt( (valid_db - query) ** 2) 
        diff[np.where(valid_db==-100)] = 0
        result = np.sum(diff, axis=1) / self.num_pitch_by_vector[valid_overlap_idx]
        # result[np.where(overlap<self.min_overlap_ratio * np.sum(nonzero_idc_query))] = 100
        # result[np.where(overlap<self.min_overlap_ratio * np.sum(self.is_nonzero, axis=1))] = 100

        mis_overlap = np.sum(np.logical_xor(nonzero_idc_query, self.is_nonzero[valid_overlap_idx]), axis=1)
        # result *= (2 - overlap/query.shape[0])
        result += mis_overlap/query.shape[0]/2

        query = np.expand_dims(query, 0)

        result_sorted = np.argsort(result)
        candidates_ids = result_sorted[:topk*10]
        filtered_result = result[candidates_ids]
        candidates_vecs = self.non_zero_db[valid_overlap_idx][candidates_ids].T
        
        cov = np.dot(query - query.mean(), candidates_vecs - candidates_vecs.mean(axis=0)) / (query.shape[1]-1)
        corr = cov / np.sqrt(np.var(query.T, ddof=1) * np.var(candidates_vecs, axis=0, ddof=1)) 
        corr = np.squeeze(corr)
        
        filtered_result -= corr / 5
        filtered_result[corr<0.5] = 100
        final_rank = np.argsort(filtered_result)
        return valid_overlap_idx[result_sorted[final_rank][:topk]], result[result_sorted[final_rank][:topk]], corr[final_rank],


        # cov = np.dot(query - query.mean(), self.non_zero_db.T - self.non_zero_db.T.mean(axis=0)) / (query.shape[1]-1)
        # corr = cov / np.sqrt(np.var(query.T, ddof=1) * np.var(self.non_zero_db.T, axis=0, ddof=1)) 
        # corr = np.squeeze(corr)
        # result -= corr / 5
        # result[corr<0.2] = 100
        # result_sorted = np.argsort(result)
        # return result_sorted[:topk], result[result_sorted[:topk]]

    
    def get_audio_by_pitch_vec_info(self, pitch_vec_index):
        pitch_vec = self.pitch_vectors[pitch_vec_index]
        song_id = pitch_vec['song_id']
        frame_pos = pitch_vec['detailed_frame_pos']
        
        id1 = int(frame_pos[0] * (self.sr / 100))
        id2 = int(frame_pos[1] * (self.sr / 100))
        
        print(song_id, frame_pos)
        return get_audio(self.audio_dir, song_id, id1, id2)


def song_id_to_pitch_txt_path(path, song_id):
    return path / str(song_id)[:3] / str(song_id)[3:6] / 'pitch_{}.txt'.format(song_id)  
        
def song_idx_to_audio_path(audio_dir, idx):
    idx = str(idx)
    path = audio_dir / idx[:3] / idx[3:6] / (idx +'.aac')
    if not path.exists():
        path = path.with_suffix('.m4a')
    return path

def get_audio(audio_dir, song_id, id1, id2):
    song_path = song_idx_to_audio_path(audio_dir, song_id)
    audio = load_audio(song_path)
    audio = audio[id1:id2]
    return audio

def load_audio(track_path, sr=44100):
    song = AudioSegment.from_file(track_path, 'm4a').set_frame_rate(sr).set_channels(1)._data
    decoded = np.frombuffer(song, dtype=np.int16) / 32768
    return decoded


def make_windowed_queries(query, hop_size, window_size, min_mel_ratio=0.6):
    num_pitch_vector = query.shape[0] // hop_size 
    c_idx = 0
    outputs = []
    for i in range(num_pitch_vector):
        if np.count_nonzero(query[c_idx:c_idx+window_size]) / window_size > min_mel_ratio:
            norm_contour = normalize_contour(np.copy(query[c_idx:c_idx+window_size]))
            if norm_contour.shape[0] < window_size:
                dummy = np.ones(window_size, dtype=float) * -100
                dummy[:norm_contour.shape[0]] = norm_contour
                norm_contour = dummy
            outputs.append(norm_contour)
        c_idx += hop_size
    return outputs

def search_candidates_from_db(queries, pitch_db, topk=20, min_overlap_ratio=0.8):
    candidates = []
    pitch_db.min_overlap_ratio = min_overlap_ratio
    # candidates = [pitch_db.search_query_from_db(contour, topk=10) for contour in tqdm(queries)]
    # candidates = [ (x,i,y,z) for i, (x,y,z) in enumerate(candidates)]
    # print(candidates)
    for i, contour in enumerate(tqdm(queries)):
        found_ids, distances, corr = pitch_db.search_query_from_db(contour, topk=50)
        candidates += [ (x, i, y, z) for x, y, z in zip(found_ids, distances, corr)]
        # candidates += found_ids[distances < 1.5].tolist()
    candidates.sort(key=lambda x:x[2])
    # candidates = list(set([ (x[0], x[2]) for x in candidates[:topk] if x[1]<10]))
    candidates = [x for x in candidates[:topk] if x[2]<10]
    return candidates

def make_query_with_different_downf(query):
    queries = []
    for down_f in [9, 10, 11]:
        ds_contour = downsample_contour(query, down_f=down_f)
        normed_query = normalize_contour(np.copy(ds_contour))
        queries.append(normed_query)
    return queries

def search_candidates_from_different_window(query, selected_contours, min_overlap_ratio=0.7):
    queries = make_query_with_different_downf(query)
    candidates = []
    total_temp_dbs= []
    for i, q in enumerate(tqdm(queries)):
        temp_db = PitchVecDB(selected_contours, '/', window_size=q.shape[0], hop_size=2)
        temp_db.min_overlap_ratio = min_overlap_ratio
        total_temp_dbs.append(temp_db)
        found_ids, distances, corr = temp_db.search_query_from_db(q, topk=10)
        candidates += [ (x, i, y, z) for x, y, z in zip(found_ids, distances, corr)]
    candidates.sort(key=lambda x:x[2])
    return candidates, queries, total_temp_dbs

def query_by_humming(query, pitch_db, metadata, meta_ids, hop_size=2, topk=20, min_mel_ratio=0.6):
    # print('updated 1')
    times = []
    frequency_in_midi = query
    times.append(time())
    queries = []
    # for i in (-1, 0, 1):
    for i in [0]:
        ds_contour = downsample_contour(frequency_in_midi, down_f=pitch_db.down_f+i, down_type='median')
        queries += make_windowed_queries(ds_contour, hop_size=hop_size, window_size=pitch_db.window_size, min_mel_ratio=min_mel_ratio)
    times.append(time())
    candidates = search_candidates_from_db(queries, pitch_db, topk=topk*3)
    times.append(time())
    # melody_candidates = list(set([pitch_db.pitch_vectors[x[0]]['melody_idx'] for x in candidates[:10]]))
    melody_candidates = list(set([pitch_db.pitch_vectors[x[0]]['melody_idx'] for x in candidates ]))
    first_song_candidates = list(set([pitch_db.pitch_vectors[x[0]]['song_id'] for x in candidates ]))
    ordered_candidates = [pitch_db.pitch_vectors[x[0]]['song_id'] for x in candidates ]
    ordered_metas = [get_meta_by_id(cand, metadata, meta_ids) for cand in ordered_candidates]
    cand_meta = [(x['track_id'], '/'.join(x['artist_name_basket']), x['track_name'], candidates[i][2], candidates[i][3])  for i, x in enumerate(ordered_metas)]
    cand_meta = delete_redundant_cand(cand_meta)
    times.append(time())

    # cand_meta = [(x['track_id'], x['artist_name_basket'], x['track_name'])  for x in metadata if x['track_id'] in first_song_candidates]

    print('Candidates in early matching:')
    for x,y,z, dist, corr in cand_meta:
        print('Artist: {}, Title: {}, ID: {}, Distance/Corr: {:.4f}/{:.4f}'.format(y,z,x, dist, corr))

    # detailed_window_size = ds_contour.shape[0]
    selected_contours = [pitch_db.contours[x] for x in melody_candidates]
    
    final_candidates, final_queries, temp_dbs = search_candidates_from_different_window(frequency_in_midi, selected_contours)
    # second_queries = make_query_with_different_downf(frequency_in_midi)
    # final_candidates = search_candidates_from_db(second_queries, temp_db, topk=10, min_overlap_ratio=0.7)
    # final_candidates = [(x[0], x[2]) for x in final_candidates[:topk]]
    final_candidates = final_candidates[:topk]
    final_song_ids = [temp_dbs[x[1]].pitch_vectors[x[0]]['song_id'] for x in final_candidates]
    final_cand_meta = [get_meta_by_id(cand, metadata, meta_ids) for cand in final_song_ids]
    final_cand_meta = [(x['track_id'], '/'.join(x['artist_name_basket']), x['track_name'], final_candidates[i][2], final_candidates[i][3])  for i, x in enumerate(final_cand_meta)]
    final_cand_meta = delete_redundant_cand(final_cand_meta)

    # final_cand_meta = [(x['track_id'], x['artist_name_basket'], x['track_name']) for x in metadata if x['track_id'] in song_ids]
    print('Candidates in final matching:')
    for x,y,z, dist, corr in final_cand_meta:
        print('Artist: {}, Title: {}, ID: {}, Distance/Corr: {:.4f}/{:.4f}'.format(y,z,x, dist, corr))

    return queries,final_queries, temp_dbs, candidates, final_candidates

    
def generate_sine_wav(melody, sr=44100, frame_rate=10):
    melody_resampled = np.repeat(melody, sr//frame_rate)
    phi = np.zeros_like(melody_resampled)
    phi[1:] = np.cumsum(2* np.pi * melody_resampled[:-1] / sr, axis=0)
    sin_wav = 0.7 * np.sin(phi)
    return sin_wav

def normalized_vec_to_orig(norm_contour, mean_pitch=60, std=5.645042479045596):
    orig = np.zeros_like(norm_contour)
    orig[norm_contour!=-100] = 440 * 2 ** ((norm_contour[norm_contour!=-100] * std + mean_pitch -69) / 12)
    orig[norm_contour==-100] = 0
    return orig 

def get_meta_by_id(id, meta, meta_ids):
    # idx = binary_index(meta_ids, id)
    idx = meta_ids.index(id)
    # print(idx)
    return meta[idx]

def delete_redundant_cand(cand_meta):
    song_id = []
    cleaned_cands = []
    for x in cand_meta:
        if x[2] not in song_id:
            song_id.append(x[2])
            cleaned_cands.append(x)
    return cleaned_cands

def midi_pitch_to_hz(alist):
    return [880 * (2 ** ((x-69)/12)) for x in alist]

# def bakset_to_str(basket):
#     if len(basket) == 1:
#         return basket[0]
#     else:
#         return ', '.join(basket)