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
from segmentation_utils import find_melody_seg_fast


class MelodyLoader:
    def __init__(self, in_midi_pitch=False):
        self.seg_thresh = 50
        self.init_seg_thresh = 1000
        self.min_mel_length = 500
        self.min_pitch_len = 10
        self.enlong_len = 10
        self.min_vocal_ratio = 0.4
        self.min_melody_ratio = 0.3
        self.max_length = 2000
        self.in_midi_pitch = in_midi_pitch
        
    def get_split_contour(self, path):
        if isinstance(path, str):
            path = Path(path)
        contour = load_melody(path)
        contour = quantizing_hz(contour, self.in_midi_pitch, self.is_quantized)
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
            self.contours = path
        self.num_neg_samples = num_neg_samples
        self.num_aug_samples = num_aug_samples
        # self.aug_types = ['different_tempo', 'different_key', 'different_std', 'addition', 'masking']
        self.aug_types = ['different_tempo', 'different_key']
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
        """
        for training:
        return: (downsampled_melody, [augmented_melodies], [negative_sampled_melodies])
        for validation:
        return: ([augmented_melodies], [selected_song_id])
        """
        selected_melody = self.contours[index]['melody']
        selected_is_vocal = self.contours[index]['is_vocal']
        selected_song_id = self.contours[index]['song_id']
        downsampled_melody = downsample_contour(selected_melody, selected_is_vocal)

        if self.set_type == 'entire':
            return downsampled_melody, selected_song_id

        aug_samples = []
        neg_samples = []
        
        # augmenting melodies
        if len(self.aug_types) <= self.num_aug_samples:
            sampled_aug_types = self.aug_types
        else:
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

        # sampling negative melodies
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

        for validation set or entire set"
        return [melodies], [song_ids]

        """
        # entire_set or valid_set with only pos_samples
        if self.num_neg == 0: 
            if self.num_pos != 0:
                out = [torch.Tensor(y) for x in batch for y in x[0]]
                song_ids = torch.LongTensor([y for x in batch for y in x[1] ])
            else:
                out = [torch.Tensor(x[0]) for x in batch]
                song_ids = torch.LongTensor([x[1] for x in batch])
            padded_sequence = torch.nn.utils.rnn.pad_sequence(out, batch_first=True)
            # input_lengths = torch.LongTensor([torch.max(padded_sequence[i, :].data.nonzero())+1 for i in range(padded_sequence.size(0))])
            input_lengths = torch.LongTensor([torch.max(torch.nonzero(padded_sequence[i, :].data))+1 for i in range(padded_sequence.size(0))])

            input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
            padded_sequence = padded_sequence[sorted_idx]
            song_ids = song_ids[sorted_idx]
            packed_data =  torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths, batch_first=True, enforce_sorted=False)
            return packed_data, song_ids

        #  training set with multiple negative sampels
        elif isinstance(batch[0][1], list): # if neg_samples are list
            total = [ [torch.Tensor(x[0])] + self.to_tensor_list(x[1]) + self.to_tensor_list(x[2]) for x in batch ]
            total_flattened = [y for x in total for y in x]
            padded_sequence = torch.nn.utils.rnn.pad_sequence(total_flattened, batch_first=True)
            # padded_sequence = padded_sequence.reshape([len(total), -1, padded_sequence.shape[1], padded_sequence.shape[2]] )
            # input_lengths = torch.LongTensor([torch.max(padded_sequence[i, :].data.nonzero())+1 for i in range(padded_sequence.size(0))])
            input_lengths = torch.LongTensor([torch.max(torch.nonzero(padded_sequence[i, :].data))+1 for i in range(padded_sequence.size(0))])
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


def hz_to_midi_pitch(hz, quantization=True):
    if hz == 0:
        return 0
    if quantization:
        return round(log2(hz/440) * 12) + 69
    else:
        return log2(hz/440) * 12 + 69


def check_song_in_genre(song, genre_list):
    for genre_id in song['genre_id_basket']:
        if genre_id in genre_list:
            return True
    return False    


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


if __name__ == '__main__':
    with open('flo_metadata.dat', 'rb') as f:
        metadata = pickle.load(f)
    # selected_genres = [29]
    selected_genres = [4]

    song_ids = get_song_ids_of_selected_genre(metadata, selected_genre=selected_genres)

    # dataset = MelodyDataset('/home/svcapp/userdata/musicai/flo_data/', song_ids=song_ids)
    # dataset.save('/home/svcapp/userdata/flo_melody/melody_kor_trot.dat')
    # pitch_path = '/home/svcapp/userdata/musicai/dev/teo/melodyExtraction_JDC/output/pitch_435845929.txt'
    # loader = MelodyLoader()
    # tokens = loader(pitch_path)

    contour_set = ContourSet('/home/svcapp/userdata/musicai/flo_data/', song_ids, quantized=False)
    contour_set.save_melody('/home/svcapp/userdata/flo_melody/contour_kor_ballade_norm.json') 