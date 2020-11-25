from math import log2
import torch
from pathlib import Path

from torch import Generator
import _pickle as pickle
import pickle as org_pickle
from tqdm import tqdm
import numpy as np
import random
import copy
import melody_augmentation as mel_aug
import json
from sampling_utils import downsample_contour


def binary_index(alist, item):
    first = 0
    last = len(alist)-1
    midpoint = 0

    if(item< alist[first]):
        return 0

    while first<last:
        midpoint = (first + last)//2
        currentElement = alist[midpoint]

        if currentElement < item:
            if alist[midpoint+1] > item:
                return midpoint
            else: first = midpoint +1
            if first == last and alist[last] > item:
                return midpoint
        elif currentElement > item:
            last = midpoint -1
        else:
            if midpoint +1 ==len(alist):
                return midpoint
            while alist[midpoint+1] == item:
                midpoint += 1
                if midpoint + 1 == len(alist):
                    return midpoint
            return midpoint
    return last


class MelodyDataset(torch.utils.data.Dataset): 
    def __init__(self, path, song_ids, pre_load=False, min_vocal_len=20):
        self.path = Path(path)
        self.melody_loader = MelodyLoader()
        self.pitch_list = [song_id_to_pitch_txt_path(self.path, x) for x in song_ids]
        print('Number of total pitch txt: {}'.format(len(self.pitch_list)))
        self.pre_load = pre_load
        if self.pre_load:
            self.load_lists_of_melody_txt()
        # self.non_vocal_index = []
        self.min_vocal_len = min_vocal_len

    def get_pitch_list(self, selected_genre):
        if selected_genre == 'entire':
            songs = [x['track_id'] for x in self.meta]
        elif isinstance(selected_genre, list):
            songs = [x['track_id'] for x in self.meta if check_song_in_genre(x, selected_genre)]        
        pitch_list = [self.path / str(x)[:3] / str(x)[3:6] / 'pitch_{}.txt'.format(x) for x in songs]
        return pitch_list
        # self.pitch_list = list(self.path.rglob('*.txt'))

    def __len__(self):
        return len(self.pitch_list) # - len(self.non_vocal_index)
    def __getitem__(self, index):
        # if index in self.non_vocal_index:
        #     return None
        if self.pre_load:
            return self.datas_in_token[index] 
        else:
            return self.melody_loader(self.pitch_list[index])
            # melody_in_token = self.melody_loader(self.pitch_list[index])
            # len_detected_pitch = sum([token['duration'] for melody in melody_in_token for token in melody['tokens']])
            # # print('Total duration of detected singing voice: {}'.format(len_detected_pitch//100))
            # if len_detected_pitch > self.min_vocal_len:
            #     return melody_in_token
            # else:
            #     self.non_vocal_index.append(index)
            #     return None

    def load_lists_of_melody_txt(self):
        self.datas_in_token = [self.melody_loader(txt) for txt in tqdm(self.pitch_list)]
        
    def save(self, out_path):
        if not self.pre_load:
            self.load_lists_of_melody_txt()
        with open(out_path, 'wb') as f:
            pickle.dump(self.datas_in_token, f)
            # np.save(self.datas_in_token, f, ensure_ascii=False)

class MelodyPreSet(torch.utils.data.Dataset):
    def __init__(self, path, set_type='train'):
        with open(path, 'rb') as f:
            self.datas_in_token = pickle.load(f)
        self.datas = [ melody for piece in self.datas_in_token for melody in piece if not melody == []]
        if set_type=='valid':
            self.datas = self.datas[int(len(self.datas) * 0.8):int(len(self.datas) * 0.9)]
        elif set_type=='train': # if train
            self.datas = self.datas[:int(len(self.datas) * 0.8)]
        elif set_type=='test':
            self.datas = self.datas[int(len(self.datas) * 0.9):]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]


class MelodyCollate:
    def __call__(self, batch):
        # batch = [x for x in batch if x is not None]
        # datas = [ (note['pitch'], duration_to_class(note['duration'])) for sample in batch for melody in sample for note in melody['tokens']]
        # return torch.LongTensor(datas) 
        notes_by_melody = [melody['tokens'] for melody in batch]
        notes_by_melody = [x for x in notes_by_melody if len(x) > 0]
        datas = [token_to_list(melody) for melody in notes_by_melody]
        # datas = [x for x in datas if x!= []]
        num_batch = len(datas)

        dummy = torch.zeros(num_batch, max([len(x) for x in datas]), 2, dtype=torch.long)
        for i in range(num_batch):
            dummy[i, :len(datas[i]), :] = torch.LongTensor(datas[i])
        return dummy
        # input_lengths = torch.LongTensor([torch.max(dummy[i, :].data.nonzero())+1 for i in range(dummy.size(0))])
        # input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
        # dummy = dummy[sorted_idx]
        # return torch.nn.utils.rnn.pack_padded_sequence(dummy, input_lengths, batch_first=True)

class MelodyLoader:
    def __init__(self, is_quantized=True, in_midi_pitch=False):
        self.q_pitch, self.q_boundary = make_quantization_info(low_pitch=110)
        self.seg_thresh = 50
        self.init_seg_thresh = 1000
        self.min_mel_length = 500
        self.min_pitch_len = 10
        self.enlong_len = 10
        self.min_vocal_ratio = 0.4
        self.min_melody_ratio = 0.3
        self.max_length = 2000
        self.is_quantized= is_quantized
        self.in_midi_pitch = in_midi_pitch
        
    def get_split_contour(self, path):
        if isinstance(path, str):
            path = Path(path)
        contour = load_melody(path)
        if self.is_quantized:
            q_contour = quantizing_hz(contour, self.in_midi_pitch, self.is_quantized)
            c_contour = clearing_note(q_contour, min_pitch_len=self.min_pitch_len)
            e_contour = elongate_note(c_contour, patience=self.enlong_len)
            melody_idxs = find_melody_segment(c_contour, self.seg_thresh, self.init_seg_thresh, min_mel_length=self.min_mel_length)
            contour = e_contour
        else:
            contour = quantizing_hz(contour, self.in_midi_pitch, self.is_quantized)
            # melody_idxs = find_melody_segment(contour, self.seg_thresh, self.init_seg_thresh, min_mel_length=self.min_mel_length)
            melody_idxs = find_melody_seg_fast(contour, self.seg_thresh, self.max_length, self.min_mel_length)
        song_len = len(contour)
        melody_len = sum([x[1]- x[0] for x in melody_idxs])
        if melody_len / song_len  > self.min_melody_ratio:
            return [{'melody': contour[x[0]:x[1]],
                    'song_id': int(path.stem[6:]),
                    'frame_pos': x
                    }
                    for x in melody_idxs]
            # return [(contour[x[0]:x[1]], int(path.stem[6:]), x) for x in melody_idxs]
        else:
            return None
        
    def __call__(self, path):
        quantized_melodies = self.get_split_contour(path)
        if quantized_melodies is not None:
            melody_in_token = [melody_to_token(melody) for melody in quantized_melodies]
            # melody_in_token = [x for x in melody_in_token if x['tokens'] != []]
            return melody_in_token
        else:
            return []




class ContourSet:
    def __init__(self, path, song_ids=[], num_aug_samples=4, num_neg_samples=4, quantized=True, pre_load=False, set_type='entire'):
        if not pre_load:
            self.path = Path(path)
            self.melody_txt_list = [song_id_to_pitch_txt_path(self.path, x) for x in song_ids]
            self.melody_loader = MelodyLoader(in_midi_pitch=True, is_quantized=quantized)

            self.contours = self.load_melody()
            self.pitch_mean, self.pitch_std = get_pitch_stat(self.contours)
            print(self.pitch_mean, self.pitch_std)
            # for i in tqdm(range(len(self.contours))):
            #     cont = self.contours[i]
            #     norm_cont = normalize_contour(cont['melody'], self.pitch_mean, self.pitch_std)
            #     self.contours[i] = {'melody':norm_cont, 'is_vocal':(np.asarray(cont['melody'])!=0).tolist(),'song_id':cont['song_id'], 'frame_pos':cont['frame_pos']}
            # norm_contours = [normalize_contour(x['melody'], self.pitch_mean, self.pitch_std) for x in contours]
            # norm_contours = [normalize_contour(x[0], self.pitch_mean, self.pitch_std) for x in self.contours]
            # self.contours = [((y, (np.asarray(x['melody'])!=0).tolist(),  x['song_id'], x['frame_pos'] for x,y in zip(self.contours, norm_contours)]
            # generator = ({'melody':y, 'is_vocal':(np.asarray(x['melody'])!=0).tolist(),'song_id':x['song_id'], 'frame_pos':x['frame_pos']} for x,y in zip(contours, norm_contours))
            # self.contours = []
            # for i in generator:
            #     self.contours.append(i)

        else:
            with open(path, 'rb') as f:
                self.contours = json.load(f)
        self.num_neg_samples = num_neg_samples
        self.num_aug_samples = num_aug_samples
        self.aug_types = ['different_tempo', 'different_key', 'different_std', 'addition', 'masking']
        self.down_f = 10
        self.set_type = set_type

        if set_type =='train':
            self.contours = self.contours[:int(len(self)*0.8)]
        elif set_type =='valid':
            self.contours = self.contours[int(len(self)*0.8):int(len(self)*0.9)]
        elif set_type == 'test':
            self.contours = self.contours[int(len(self)*0.9):]

    def load_melody(self):
        # melody_txt_list = self.path.rglob('*.txt')
        contours = [self.melody_loader.get_split_contour(txt) for txt in tqdm(self.melody_txt_list)]
        contours = [x for x in contours if x is not None]
        contours = [y for x in contours for y in x]
        return contours
    
    def save_melody(self, out_path):
        # contours = self.load_melody()
        with open(out_path, 'w') as f:
            json.dump(self.contours, f)

    def __len__(self):
        return len(self.contours)

    def __getitem__(self, index):
        selected_melody = self.contours[index]['melody']
        selected_is_vocal = self.contours[index]['is_vocal']
        selected_song_id = self.contours[index]['song_id']
        downsampled_melody = downsample_contour(selected_melody, selected_is_vocal)

        if self.set_type == 'entire':
            return downsampled_melody, selected_song_id

        aug_samples = []
        neg_samples = []
        
        sampled_aug_types = random.sample(self.aug_types, self.num_aug_samples)
        for aug_type in sampled_aug_types:
            if aug_type == 'different_tempo':
                aug_melody = getattr(mel_aug, 'with_different_tempo')(selected_melody, selected_is_vocal)
            else:
                func = getattr(mel_aug, 'with_'+aug_type)
                aug_melody = func(downsampled_melody)
            aug_samples.append(aug_melody)
        
        if self.set_type == 'valid':
            return aug_samples, [selected_song_id] * len(aug_samples)

        while len(neg_samples) < self.num_neg_samples:
            neg_idx = random.randint(0, len(self)-1)
            if self.contours[neg_idx]['song_id'] != selected_song_id:
                neg_samples.append(downsample_contour(self.contours[neg_idx]['melody'], self.contours[neg_idx]['is_vocal'], self.down_f))
        return downsampled_melody, aug_samples, neg_samples


def pad_collate(batch):
    seq = [torch.Tensor(x) for x in batch]
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)


class ContourCollate:
    # def __init__(self, mean, std):
    #     self.mean = mean
    #     self.std = std
    def __init__(self, num_pos, num_neg):
        self.num_pos = num_pos
        self.num_neg = num_neg

    def to_tensor_list(self, alist):
        return [torch.Tensor(x) for x in alist]

    def __call__(self, batch):
        """Collate's training batch from melody
        ------
        batch: list of [anchor_mel, [pos_mels], [neg_mels]]
        """
        if self.num_neg == 0: # entire_set or valid_set with only pos_samples
            if self.num_pos != 0:
                out = [torch.Tensor(y) for x in batch for y in x[0]]
                song_ids = torch.LongTensor([y for x in batch for y in x[1] ])
            else:
                out = [torch.Tensor(x[0]) for x in batch]
                song_ids = torch.LongTensor([x[1] for x in batch])
            padded_sequence = torch.nn.utils.rnn.pad_sequence(out, batch_first=True)
            input_lengths = torch.LongTensor([torch.max(padded_sequence[i, :].data.nonzero())+1 for i in range(padded_sequence.size(0))])
            input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
            padded_sequence = padded_sequence[sorted_idx]
            song_ids = song_ids[sorted_idx]
            packed_data =  torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths, batch_first=True, enforce_sorted=False)
            return packed_data, song_ids

        if isinstance(batch[0][1], list): # if neg_mels are list
            total = [ [torch.Tensor(x[0])] + self.to_tensor_list(x[1]) + self.to_tensor_list(x[2]) for x in batch ]
            total_flattend = [y for x in total for y in x]
            padded_sequence = torch.nn.utils.rnn.pad_sequence(total_flattend, batch_first=True)
            # padded_sequence = padded_sequence.reshape([len(total), -1, padded_sequence.shape[1], padded_sequence.shape[2]] )
            input_lengths = torch.LongTensor([torch.max(padded_sequence[i, :].data.nonzero())+1 for i in range(padded_sequence.size(0))])
            input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
            padded_sequence = padded_sequence[sorted_idx]
            packed_data =  torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths, batch_first=True, enforce_sorted=False)
            return packed_data

        else:
            dummy = np.zeros((len(batch), 3, self.mel_bins, self.mel_length) )
            for i, sample in enumerate(batch):
                for j in range(3):
                    dummy[i, j, :, :sample[j].shape[1]] = sample[j]
            mels = torch.FloatTensor(dummy)
            return mels[:, 0, :, :], mels[:, 1, :, :], mels[:, 2, :, :]

def find_melody_segment(contour, threshold=50, init_thresh=1000, min_vocal_ratio=0.4, min_mel_length=500):
    melody_idxs = []
    continuing = False
    non_pitch_count = 0
    pitch_count = 0
    current_slice_idx = [0, 0]
    for i in range(len(contour)):
        pitch = contour[i]
        if pitch > 0:
            non_pitch_count = 0
            if not continuing:
                continuing = True
                current_slice_idx[0] = i
                pitch_count += 1
            else:
                pitch_count += 1
        else:
            if continuing:
                non_pitch_count += 1
            melody_duration = i-current_slice_idx[0]
            if non_pitch_count > threshold + max(0, init_thresh-melody_duration):
                current_slice_idx[1] = i
                if current_slice_idx[0] != 0 and pitch_count / melody_duration > min_vocal_ratio and melody_duration > min_mel_length:
                    melody_idxs.append(current_slice_idx)
                current_slice_idx = [0, 0]
                non_pitch_count= 0
                pitch_count = 0
                continuing = False
    return melody_idxs

def find_melody_seg_fast(contour,zero_threshold, max_length, min_length):
    zeros_slice = get_zero_slice_from_contour(contour, threshold=zero_threshold)
    voice = zero_slice_to_segment(zeros_slice)
    if voice != []:
        expand_voice(voice, max_length=max_length)
    voice = [(int(x[0]), int(x[1])) for x in voice if x[1]-x[0]>min_length]
    return voice

def elongate_note(q_contour, patience=10):
    output = []
    prev_pitch = 0
    non_pitch_count = 0
    for pitch in q_contour:
        if pitch > 0:
            output.append(pitch)
            prev_pitch = pitch
            non_pitch_count = 0
        else:
            non_pitch_count += 1
            if non_pitch_count > patience:
                prev_pitch = 0
                non_pitch_count = 0
            output.append(prev_pitch)
    return output

def clearing_note(q_contour, min_pitch_len=5):
    prev_pitch = 0
    prev_pitch_start = 0
    output = [x for x in q_contour]
    for i in range(len(q_contour)):
        pitch = q_contour[i]
        if pitch != prev_pitch:
            prev_pitch_duration = i - prev_pitch_start
            if prev_pitch_duration < min_pitch_len:
                output[prev_pitch_start:i] = [0] * prev_pitch_duration
            prev_pitch = pitch
            prev_pitch_start = i
    return output

def quantizing_hz(contour, to_midi=False, quantization=True):
    if quantization is False and to_midi is False:
        return contour
    def quantize_or_return_zero(pitch):
        if pitch > 0:
            if to_midi:
                return hz_to_midi_pitch(pitch, quantization)
            else:
                return 440 * (2 ** ((round(log2(pitch/440) * 12))/12))
        else:
            return 0
    return [quantize_or_return_zero(x) for x in contour]

def load_melody(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [float(x.split(' ')[1][:-2]) for x in lines]

def make_quantization_info(low_pitch=110):
    pitch = low_pitch
    quantized_pitch = [pitch]
    quantized_pitch_boundary = [pitch * (2 ** (1/24)) ]
    for i in range(48):
        pitch *= 2 ** (1/12)
        quantized_pitch.append(pitch)
        quantized_pitch_boundary.append(quantized_pitch_boundary[-1] * 2 ** (1/12))
    return quantized_pitch, quantized_pitch_boundary

def hz_to_midi_pitch(hz, quantization=True):
    if hz == 0:
        return 0
    if quantization:
        return round(log2(hz/440) * 12) + 69
    else:
        return log2(hz/440) * 12 + 69

def melody_to_token(melody, min_pitch_length=10):
    tokens = {'tokens':[], 'frame_pos':melody['frame_pos'], 'song_id':melody['song_id']}
    prev_pitch = melody['melody'][0]
    pitch_duration = 0
    for pitch in melody['melody']:
        if pitch == prev_pitch:
            pitch_duration += 1
        else:
            # if pitch_duration > min_pitch_length:
            #     tokens['tokens'].append({'pitch': hz_to_midi_pitch(prev_pitch), 'duration': pitch_duration})
            # else:
            #     print(pitch, pitch_duration)
            tokens['tokens'].append({'pitch': hz_to_midi_pitch(prev_pitch), 'duration': pitch_duration})
            pitch_duration = 1
            prev_pitch = pitch
    tokens['tokens'].append({'pitch': hz_to_midi_pitch(prev_pitch), 'duration': pitch_duration})
    return tokens

def token_to_list(in_seq):
    # [ (in_seq[i][0] - in_seq[i], in_seq[i][1]) for i in range(1, len(in_seq))]
    # return [ (note['pitch'] - 44, duration_to_class(note['duration'])) for note in in_seq ]
    filtered_seq = [note for note in in_seq if note['pitch']!=0]
    seq = [ (note['pitch'], duration_to_class(note['duration'])) for note in filtered_seq ]
    init = [(50, seq[0][1])]
    init += [(seq[i][0]- seq[i-1][0]  + 50, seq[i][1]) for i in range(1,len(filtered_seq))]
    return init

def check_song_in_genre(song, genre_list):
    for genre_id in song['genre_id_basket']:
        if genre_id in genre_list:
            return True
    return False    


def duration_to_class(duration):
    if duration < 20:
        return 0
    elif duration < 50:
        return 1
    elif duration < 100:
        return 2
    elif duration < 200:
        return 3
    elif duration < 400:
        return 4
    else:
        return 5

def cal_total_voice_sec(melody_in_token):
    return sum([token['duration'] for melody in melody_in_token for token in melody['tokens']])

def get_song_ids_of_selected_genre(meta, selected_genre):
    if selected_genre == 'entire':
        songs = [x['track_id'] for x in meta]
    elif isinstance(selected_genre, list):
        songs = [x['track_id'] for x in meta if check_song_in_genre(x, selected_genre)]       
    return songs

def normalize_contour(contour, mean, std):
    return [normalize_value_if_not_zero(x, mean, std) for x in contour]

def normalize_value_if_not_zero(value, mean, std):
    if value == 0:
        return value
    else:
        return (value-mean) / std

def get_pitch_stat(contours):
    pitch_values = [y for x in contours for y in x['melody'] if y!=0]
    mean = np.mean(pitch_values)
    std = np.std(pitch_values)
    return mean, std

def song_id_to_pitch_txt_path(path, song_id):
    # path: pathlib.Path()
    return path / str(song_id)[:3] / str(song_id)[3:6] / 'pitch_{}.txt'.format(song_id)

def get_zero_slice_from_contour(contour, threshold=50):
    contour_array = np.asarray(contour)
    is_zero_position = np.where(contour_array == 0)[0]
    diff_by_position = np.diff(is_zero_position)
    slice_pos = np.where(diff_by_position>1)[0]
    voice_frame = np.stack([is_zero_position[slice_pos]+1, is_zero_position[slice_pos] + diff_by_position[slice_pos]], axis=-1)
    if voice_frame.shape[0] == 0:
        zeros_slice = []
    else:
        zeros_slice = [ [0, voice_frame[0,0]] ] + [ [voice_frame[i-1,1], voice_frame[i,0]] for i in range(1, voice_frame.shape[0])]
        zeros_slice = [x for x in zeros_slice if x[1]-x[0] > threshold]
    return zeros_slice

def zero_slice_to_segment(zeros_slice, min_voice_seg=5):
    return [ (zeros_slice[i][1], zeros_slice[i+1][0]) for i in range(len(zeros_slice)-1) if  (zeros_slice[i+1][0] - zeros_slice[i][1]) >= min_voice_seg]

def expand_voice(voice_slice, max_length=2000):
    def merged_length(alist, idx):
        return alist[idx][0] + alist[idx][1] + alist[idx+1][0]
    len_and_distance = get_length_and_distance_of_melody(voice_slice)
#     valid_distances = [len_and_distance[i][1] for i in range(len(len_and_distance)-1) if len_and_distance[i][0] +len_and_distance[i+1][0]<max_length]
    valid_distances = [ len_and_distance[i][1] for i in range(len(len_and_distance)-1) if merged_length(len_and_distance, i) <max_length]
    while valid_distances:
        min_distance = min(valid_distances)
        min_index = [i for i in range(len(len_and_distance)-1) if len_and_distance[i][1] ==min_distance and  merged_length(len_and_distance, i) <max_length]
        for index in reversed(min_index):
            merge_voice_slice(voice_slice, index)
        if voice_slice == []:
            valid_distances = []
        else:
            len_and_distance = get_length_and_distance_of_melody(voice_slice)
            valid_distances = [ len_and_distance[i][1] for i in range(len(len_and_distance)-1) if merged_length(len_and_distance, i) <max_length]
    return voice_slice

def merge_voice_slice(voice_slice, index):
    first = voice_slice.pop(index)
    second = voice_slice.pop(index)
    new = (first[0], second[1])
    voice_slice.insert(index, new)

def get_length_and_distance_of_melody(voice_slice):
    return [ (voice_slice[i][1]-voice_slice[i][0], voice_slice[i+1][0]-voice_slice[i][1]) for i in range(len(voice_slice)-1)] + [(voice_slice[-1][1]-voice_slice[-1][0], 10000 )]


if __name__ == '__main__':
    dataset = MelodyDataset('/home/svcapp/userdata/musicai/flo_data/')
    dataset.save('/home/svcapp/userdata/flo_melody/melody_entire.dat')
    # pitch_path = '/home/svcapp/userdata/musicai/dev/teo/melodyExtraction_JDC/output/pitch_435845929.txt'
    # loader = MelodyLoader()
    # tokens = loader(pitch_path)

