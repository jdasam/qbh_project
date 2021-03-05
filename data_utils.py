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
from sampling_utils import downsample_contour, downsample_contour_array
from segmentation_utils import find_melody_seg_fast
from melody_utils import MelodyLoader
import librosa
from madmom.audio.signal import Signal
# from pydub import AudioSegment

class ContourSet:
    def __init__(self, path, song_ids=[], num_aug_samples=4, num_neg_samples=4, quantized=True, pre_load=False, set_type='entire', min_aug=1):
        if not pre_load:
            self.path = Path(path)
            self.melody_txt_list = [song_id_to_pitch_txt_path(self.path, x) for x in song_ids]
            self.melody_loader = MelodyLoader(in_midi_pitch=True, is_quantized=quantized)

            self.contours = self.load_melody()
            # self.pitch_mean, self.pitch_std = get_pitch_stat(self.contours)
            # print(self.pitch_mean, self.pitch_std)
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
        self.aug_keys = ['tempo', 'key', 'std', 'masking', 'pitch_noise', 'fill', 'drop_out']
        # self.aug_types = ['different_tempo', 'different_key']
        self.down_f = 10
        self.set_type = set_type
        self.min_aug = min_aug

        if set_type =='train':
            self.contours = self.contours[:int(len(self)*0.8)]
        elif set_type =='valid':
            self.contours = self.contours[int(len(self)*0.8):int(len(self)*0.9)]
        elif set_type == 'test':
            self.contours = self.contours[int(len(self)*0.9):]

    def load_melody(self):
        # melody_txt_list = self.path.rglob('*.txt')
        # contours = [self.melody_loader.get_split_contour(txt) for txt in tqdm(self.melody_txt_list)]
        contours = [self.melody_loader.get_overlapped_contours(txt) for txt in tqdm(self.melody_txt_list)]
        contours = [x for x in contours if x is not None]
        contours = [y for x in contours for y in x]
        return contours
    
    def save_melody(self, out_path):
        # contours = self.load_melody()
        # with open(out_path, 'w') as f:
        #     json.dump(self.contours, f)
        with open(out_path, 'wb') as f:
            pickle.dump(self.contours, f)

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
        melody_array = mel_aug.melody_dict_to_array(self.contours[index])
        if self.min_aug < len(self.aug_keys):
            aug_samples = [mel_aug.make_augmented_melody(melody_array, random.sample(self.aug_keys, random.randint(self.min_aug,len(self.aug_keys)))) for i in range(self.num_aug_samples)]
        else:
            aug_samples = [mel_aug.make_augmented_melody(melody_array,self.aug_keys) for i in range(self.num_aug_samples)]
        # if len(self.aug_types) <= self.num_aug_samples:
        #     sampled_aug_types = self.aug_types
        # else:
        #     sampled_aug_types = random.sample(self.aug_types, self.num_aug_samples)
        # for aug_type in sampled_aug_types:
        #     if aug_type == 'different_tempo':
        #         aug_melody = getattr(mel_aug, 'with_different_tempo')(selected_melody, selected_is_vocal)
        #     else:
        #         func = getattr(mel_aug, 'with_'+aug_type)
        #         aug_melody = func(downsampled_melody)
        #     aug_samples.append(aug_melody)
        
        if self.set_type == 'valid':
            return aug_samples, [selected_song_id] * len(aug_samples)
            # return [downsampled_melody] * len(aug_samples), [selected_song_id] * len(aug_samples)

        # sampling negative melodies
        while len(neg_samples) < self.num_neg_samples:
            neg_idx = random.randint(0, len(self)-1)
            if self.contours[neg_idx]['song_id'] != selected_song_id:
                neg_samples.append(downsample_contour(self.contours[neg_idx]['melody'], self.contours[neg_idx]['is_vocal'], self.down_f))
        return downsampled_melody, aug_samples, neg_samples



class WindowedContourSet:
    def __init__(self, path, aug_weights, song_ids=[], num_aug_samples=4, num_neg_samples=4, quantized=True, pre_load=False, pre_load_data=None, set_type='entire', min_aug=1, min_vocal_ratio=0.5):
        self.min_vocal_ratio = min_vocal_ratio
        self.path = Path(path)
        if not pre_load:
            self.melody_txt_list = [song_id_to_pitch_txt_path(self.path, x) for x in song_ids]
            self.melody_loader = MelodyLoader(in_midi_pitch=True, is_quantized=quantized)
            self.contours = self.load_melody()
        else:
            self.contours = pre_load_data
        self.num_neg_samples = num_neg_samples
        self.num_aug_samples = num_aug_samples
        self.aug_keys = ['tempo', 'key', 'std', 'masking', 'pitch_noise', 'fill', 'smoothing', 'absurd_noise']
        # self.aug_types = ['different_tempo', 'different_key']
        self.down_f = 10
        self.set_type = set_type
        self.min_aug = min_aug

        if set_type =='train':
            # self.contours = self.contours[:int(len(self)*0.8)]
            self.contours = self.contours
        elif set_type =='valid':
            self.contours = self.contours[int(len(self)*0.8):int(len(self)*0.9)]
        elif set_type == 'test':
            self.contours = self.contours[int(len(self)*0.9):]
        
        if aug_weights != []:
            self.melody_augmentor = mel_aug.MelodyAugmenter(aug_weights)

    def load_melody(self):
        contours = [self.melody_loader.get_overlapped_contours(txt, min_ratio=self.min_vocal_ratio) for txt in tqdm(self.melody_txt_list)]
        contours = [x for x in contours if x is not None and x != []]
        contours = [y for x in contours for y in x]
        return contours
    
    def save_melody(self, out_path):
        with open(out_path, 'wb') as f:
            pickle.dump(self.contours, f)

    def __len__(self):
        return len(self.contours)

    def __getitem__(self, index):
        """
        for training:
        return: (downsampled_melody, [augmented_melodies], [negative_sampled_melodies])
        for validation:
        return: ([augmented_melodies], [selected_song_id])
        """
        selected_melody = self.contours[index]['contour']
        selected_song_id = self.contours[index]['song_id']
        downsampled_melody = downsample_contour_array(selected_melody)

        if self.set_type == 'entire':
            return downsampled_melody, selected_song_id

        aug_samples = []
        neg_samples = []
        
        if self.min_aug < len(self.aug_keys):
            aug_samples = [self.melody_augmentor(selected_melody, random.sample(self.aug_keys, random.randint(self.min_aug,len(self.aug_keys)))) for i in range(self.num_aug_samples)]
        else:
            aug_samples = [self.melody_augmentor(selected_melody, self.aug_keys) for i in range(self.num_aug_samples)]
        
        if self.set_type == 'valid':
            return aug_samples, [selected_song_id] * len(aug_samples)
            # return [downsampled_melody] * len(aug_samples), [selected_song_id] * len(aug_samples)

        # sampling negative melodies
        while len(neg_samples) < self.num_neg_samples:
            neg_idx = random.randint(0, len(self)-1)
            if self.contours[neg_idx]['song_id'] != selected_song_id:
                neg_samples.append(downsample_contour_array(self.contours[neg_idx]['contour'], self.down_f))
        return downsampled_melody, aug_samples, neg_samples


class HummingPairSet:
    def __init__(self, contour_pairs, aug_weights, set_type, aug_keys, num_aug_samples=4, num_neg_samples=4 ):
        # with open(path, "rb") as f:
        #     self.contour_pairs = pickle.load(f)
        self.contours = contour_pairs
        self.num_neg_samples = num_neg_samples
        self.num_aug_samples = num_aug_samples
        self.aug_keys = aug_keys
        self.set_type = set_type
        self.down_f = 10

        if aug_weights != []:
            self.melody_augmentor = mel_aug.MelodyAugmenter(aug_weights)
            
        assert len(self.contours) == 1400
            
        if set_type =='train':
            self.contours = [x for x in self.contours if x['meta']['song_group']=="900"]
        elif set_type =='valid':
            # self.contours = [x for x in self.contours if x['meta']['song_group']=="100" and x['meta']['singer_group']=="P"]
            self.contours = [x for x in self.contours if x['meta']['song_group']=="100" and int(x['meta']['song_idx'])%2==0]
        elif set_type == 'test':
            # self.contours = [x for x in self.contours if x['meta']['song_group']=="100" and x['meta']['singer_group']=="N"]
            self.contours = [x for x in self.contours if x['meta']['song_group']=="100" and int(x['meta']['song_idx'])%2==1]

    def __getitem__(self, index):
        """
        for training:
        return: (downsampled_melody, [augmented_melodies], [negative_sampled_melodies])
        for validation:
        return: ([augmented_melodies], [selected_song_id])
        """
        selected_melody = self.contours[index]['humm']
        original_melody = self.contours[index]['orig']
        selected_song_id = self.contours[index]['meta']['track_id']
        orig_ds_melody = downsample_contour_array(original_melody)

        aug_samples = []
        neg_samples = []
        
        
        if self.set_type == 'valid' or self.set_type == 'test':
            downsampled_melody = downsample_contour_array(selected_melody)
            return downsampled_melody, selected_song_id
            # return [downsampled_melody] * len(aug_samples), [selected_song_id] * len(aug_samples)
        
        aug_samples = [self.melody_augmentor(selected_melody,self.aug_keys) for i in range(self.num_aug_samples)]
        
        # sampling negative melodies
        while len(neg_samples) < self.num_neg_samples:
            neg_idx = random.randint(0, len(self)-1)
            if self.contours[neg_idx]['meta']['track_id'] != selected_song_id:
                neg_samples.append(downsample_contour_array(self.contours[neg_idx]['humm'], self.down_f))
        return orig_ds_melody, aug_samples, neg_samples

    def __len__(self):
        return len(self.contours)


def pad_collate(batch):
    seq = [torch.Tensor(x) for x in batch]
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)


class AudioSet(WindowedContourSet):
    def __init__(self, path, aug_weights, song_ids=[], num_aug_samples=4, num_neg_samples=4, quantized=True, pre_load=False, pre_load_data=None, set_type='entire', min_aug=1, min_vocal_ratio=0.5):
        super(AudioSet, self).__init__(path, aug_weights, song_ids, num_aug_samples, num_neg_samples, quantized, pre_load, pre_load_data, set_type, min_aug, min_vocal_ratio)
        self.x_train_mean = np.load('x_data_mean_total_31.npy')
        self.x_train_std = np.load('x_data_std_total_31.npy')

    def load_audio(self, song_id, frame_pos):
        audio_path = song_id_to_audio_path(self.path, song_id)
        x_test, x_spec = load_audio_and_get_spec(audio_path, frame_pos)
        x_test = (x_test-self.x_train_mean)/(self.x_train_std+0.0001)
        x_test = x_test[:, :, :, np.newaxis]
        return x_test
    

    def __getitem__(self, index):
        """
        for training:
        return: (downsampled_melody, [augmented_melodies], [negative_sampled_melodies])
        for validation:
        return: ([augmented_melodies], [selected_song_id])
        """
        selected_melody = self.contours[index]['contour']
        selected_song_id = self.contours[index]['song_id']
        selected_frame = self.contours[index]['frame_pos']
        # downsampled_melody = downsample_contour_array(selected_melody)
        if self.set_type != "valid":
            original_audio = self.load_audio(selected_song_id, selected_frame)

        if self.set_type == 'entire':
            return original_audio, selected_song_id

        aug_samples = []
        neg_samples = []
        
        if self.min_aug < len(self.aug_keys):
            aug_samples = [self.melody_augmentor(selected_melody, random.sample(self.aug_keys, random.randint(self.min_aug,len(self.aug_keys)))) for i in range(self.num_aug_samples)]
        else:
            aug_samples = [self.melody_augmentor(selected_melody, self.aug_keys) for i in range(self.num_aug_samples)]
        
        if self.set_type == 'valid':
            return aug_samples, [selected_song_id] * len(aug_samples)
            # return [downsampled_melody] * len(aug_samples), [selected_song_id] * len(aug_samples)

        # sampling negative melodies
        while len(neg_samples) < self.num_neg_samples:
            neg_idx = random.randint(0, len(self)-1)
            if self.contours[neg_idx]['song_id'] != selected_song_id:
                neg_samples.append(downsample_contour_array(self.contours[neg_idx]['contour'], self.down_f))
        return original_audio, aug_samples, neg_samples

        

class ContourCollate:
    # def __init__(self, mean, std):
    #     self.mean = mean
    #     self.std = std
    def __init__(self, num_pos, num_neg, for_cnn=False):
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.for_cnn = for_cnn

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

            if self.for_cnn:
                max_length = max([len(x) for x in out])
                dummy = torch.zeros(len(out), max_length, 2)
                for i in range(len(out)):
                    seq = out[i]
                    left_margin = (max_length - seq.shape[0]) // 2
                    dummy[i,left_margin:left_margin+seq.shape[0],:] = seq
                return dummy, song_ids
            else:
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
            if self.for_cnn:
                max_length = max([len(x) for x in total_flattened])
                dummy = torch.zeros(len(total_flattened), max_length, 2)
                for i in range(len(total_flattened)):
                    seq = total_flattened[i]
                    left_margin = (max_length - seq.shape[0]) // 2
                    dummy[i,left_margin:left_margin+seq.shape[0],:] = seq
                return dummy 
            else:
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


class AudioContourCollate:
    # def __init__(self, num_pos, num_neg, for_cnn=False):
    #     self.num_pos = num_pos
    #     self.num_neg = num_neg

    def to_tensor_list(self, alist):
        return [torch.Tensor(x) for x in alist]

    def __call__(self, batch):
        # batch: [(audio_anchor, positive_contour, negative_contour) ]* num_batch
        anchor_audio = torch.Tensor([x[0] for x in batch])
        anchor_audio.requires_grad = False
        total = [self.to_tensor_list(x[1]) + self.to_tensor_list(x[2]) for x in batch ]
        total_flattened = [y for x in total for y in x]
        max_length = max([len(x) for x in total_flattened])
        dummy = torch.zeros(len(total_flattened), max_length, 2)
        for i in range(len(total_flattened)):
            seq = total_flattened[i]
            left_margin = (max_length - seq.shape[0]) // 2
            dummy[i,left_margin:left_margin+seq.shape[0],:] = seq
        return anchor_audio, dummy

class AudioCollate:
    def __call__(self, batch):
        out = [torch.Tensor(x[0]) for x in batch]
        song_ids = torch.LongTensor([x[1] for x in batch])

        max_length = max([len(x) for x in out])
        dummy = torch.zeros(len(out), max_length, out[0].shape[1], out[0].shape[2], 1)
        for i in range(len(out)):
            seq = out[i]
            left_margin = (max_length - seq.shape[0]) // 2
            dummy[i,left_margin:left_margin+seq.shape[0]] = seq
        return dummy, song_ids


class AudioOnlySet:
    def __init__(self, path, song_ids,  set_type='entire'):
        self.path = Path(path)
        self.songs = song_ids
        self.x_train_mean = np.load('x_data_mean_total_31.npy')
        self.x_train_std = np.load('x_data_std_total_31.npy')
        self.slice_sec = 3
        self.sample_rate = 8000
        self.slice_samples = self.slice_sec * self.sample_rate
        self.win_size = 31

        if set_type =='train':
            # self.contours = self.contours[:int(len(self)*0.8)]
            self.songs = self.songs[:int(len(self)*0.8)]
        elif set_type =='valid':
            self.songs = self.songs[int(len(self)*0.8):int(len(self)*0.9)]
        elif set_type == 'test':
            self.songs = self.songs[int(len(self)*0.9):]

    # def load_audio(self, song_id, frame_pos):
    #     audio_path = song_id_to_audio_path(self.path, song_id)
    #     x_test, x_spec = load_audio_and_get_spec(audio_path, frame_pos)
    #     x_test = (x_test-self.x_train_mean)/(self.x_train_std+0.0001)
    #     x_test = x_test[:, :, :, np.newaxis]
    #     return x_test
    
    def spec_to_slice_norm(self, spec, win_size):
        num_frames = spec.shape[1]

        # for padding
        padNum = num_frames % win_size
        if padNum != 0:
            len_pad = win_size - padNum
            padding_feature = np.zeros(shape=(513, len_pad))
            spec = np.concatenate((spec, padding_feature), axis=1)
            num_frames = num_frames + len_pad
        
        x_test = spec.reshape(spec.shape[0], 1, -1, win_size).transpose(2, 1, 3,0)
        x_test = (x_test-self.x_train_mean)/(self.x_train_std+0.0001)
        return x_test

    def __len__(self):
        return len(self.songs)


    def __getitem__(self, index):
        """
        for training:
        return: (downsampled_melody, [augmented_melodies], [negative_sampled_melodies])
        for validation:
        return: ([augmented_melodies], [selected_song_id])
        """
        selected_song = self.songs[index]
        # downsampled_melody = downsample_contour_array(selected_melody)
        audio_path = song_id_to_audio_path(self.path, selected_song)
        audio_samples = load_audio_sample(audio_path, self.sample_rate)
        slice_pos = random.randint(0, len(audio_samples)-1-self.slice_samples)
        audio_sliced = audio_samples[slice_pos:slice_pos + self.slice_samples]
        
        spec_10 = get_spec_with_librosa(audio_sliced)
        spec_10 = self.spec_to_slice_norm(spec_10, self.win_size)
        return audio_sliced, spec_10

class AudioSpecCollate:
    def __call__(self, batch):
        audio = torch.Tensor([x[0] for x in batch])
        spec = torch.Tensor([x[1] for x in batch])
        spec = spec.view(spec.shape[0]*spec.shape[1], 1, spec.shape[3], spec.shape[4])
        return audio, spec

# class HummingData:
#     def __init__(self, path):
#         selected_100, selected_900 = humm_utils.load_meta_from_excel()
#         self.humming_db = humm_utils.HummingDB('/home/svcapp/userdata/humming_db', '/home/svcapp/userdata/flo_data_backup/', selected_100, selected_900)
#         self.contour_pitch = [x for x in self.humming_db]
        
    

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
    txt_path = path / str(song_id)[:3] / str(song_id)[3:6] / '{}_pitch.txt'.format(song_id)
    # txt_path = path / str(song_id)[:3] / str(song_id)[3:6] / 'pitch_{}.txt'.format(song_id)
    if not txt_path.exists():
        txt_path = path / 'qbh' / f'{song_id}_pitch.txt'
    return txt_path
    # return path  / f'pitch_{song_id}.txt'

def song_id_to_audio_path(path, song_id):
    # path: pathlib.Path()
    audio_path = path / str(song_id)[:3] / str(song_id)[3:6] / '{}.aac'.format(song_id)
    if not audio_path.exists():
        audio_path = audio_path.with_suffix('.m4a')
    if not audio_path.exists():
        audio_path = path / 'qbh' / f'{song_id}.aac'
    return audio_path
    # return path  / f'pitch_{song_id}.txt'


def load_audio_sample(path, sr=8000):
    y = Signal(str(path), sample_rate=sr, dtype=np.float32, num_channels=1)
    return y

def get_spec_with_librosa(y):
    S = librosa.core.stft(y, n_fft=1024, hop_length=80*1, win_length=1024)
    x_spec = np.abs(S)
    x_spec = librosa.core.power_to_db(x_spec, ref=np.max)
    x_spec = x_spec.astype(np.float32)
    return x_spec

def load_audio_and_get_spec(path, frame_pos, win_size=31):
    x_test = []

    # y, sr = librosa.load(file_name, sr=8000)
    # *********** madmom.Signal() is faster than librosa.load() ***********
    # audio = AudioSegment.from_file(path, "m4a").set_frame_rate(8000).set_channels(1)._data
    # y = np.frombuffer(audio, dtype=np.int16) / 32768
    y = Signal(path, sample_rate=8000, dtype=np.float32, num_channels=1)
    y_slice = y[frame_pos[0]*80:frame_pos[1]*80]
    S = librosa.core.stft(y_slice, n_fft=1024, hop_length=80*1, win_length=1024)
    x_spec = np.abs(S)
    x_spec = librosa.core.power_to_db(x_spec, ref=np.max)
    x_spec = x_spec.astype(np.float32)
    num_frames = x_spec.shape[1]

    # for padding
    padNum = num_frames % win_size
    if padNum != 0:
        len_pad = win_size - padNum
        padding_feature = np.zeros(shape=(513, len_pad))
        x_spec = np.concatenate((x_spec, padding_feature), axis=1)
        num_frames = num_frames + len_pad

    for j in range(0, num_frames, win_size):
        x_test_tmp = x_spec[:, range(j, j + win_size)].T
        x_test.append(x_test_tmp)
    x_test = np.array(x_test)

    return x_test, x_spec

if __name__ == '__main__':
    with open('flo_metadata.dat', 'rb') as f:
        metadata = pickle.load(f)
    # selected_genres = [29]
    # selected_genres = [4]
    selected_genres = [4, 12, 13, 17, 10, 7,15, 11, 9]

    song_ids = get_song_ids_of_selected_genre(metadata, selected_genre=selected_genres)
    with open('humm_db_ids.dat', 'rb') as f:
        humm_ids = pickle.load(f)
    song_ids += humm_ids
    # qbh_path = Path('/home/svcapp/userdata/flo_data_backup/qbh')
    # song_ids += [int(x.stem)for x in qbh_path.rglob("*.aac")]
    
    # dataset = MelodyDataset('/home/svcapp/userdata/musicai/flo_data/', song_ids=song_ids)
    # dataset.save('/home/svcapp/userdata/flo_melody/melody_kor_trot.dat')
    # pitch_path = '/home/svcapp/userdata/musicai/dev/teo/melodyExtraction_JDC/output/pitch_435845929.txt'
    # loader = MelodyLoader()
    # tokens = loader(pitch_path)

    contour_set = WindowedContourSet('/home/svcapp/userdata/flo_data_backup/', song_ids, quantized=False)
    contour_set.save_melody('/home/svcapp/userdata/flo_melody/overlapped.dat') 

# 61.57743176094967 5.62532396823295
# (61.702336487738215, 5.5201786930065415) for '/home/svcapp/userdata/flo_melody/contour_subgenre_norm.json'

'''
해외 메탈 18
국내 발라드 4
해외 힙합 14
기타 30
클래식 20
재즈 22
국내 락/메탈 12
해외 팝 13
월드뮤직 26
해외 락 17
J-POP 25
뉴에이지 21
국내 알앤비 7
CCM 28
해외 알앤비 15
맘/태교 24
OST/BGM 19
국내 댄스/일렉 5
해외 일렉트로닉 16
트로트 8
국내 포크/블루스 11
키즈 23
국내 인디 9
종교음악 29
국악 27
국내 팝/어쿠스틱 10
국내 힙합 6
'''