from math import log2
import torch
from pathlib import Path

import _pickle as pickle
from tqdm import tqdm
import numpy as np
import random
import copy
import melody_augmentation as mel_aug
from sampling_utils import downsample_contour_array
from melody_utils import MelodyLoader
        

class WindowedContourSet:
    def __init__(self, path, aug_weights, song_ids=[], num_aug_samples=4, num_neg_samples=4, quantized=True, pre_load=False, set_type='entire', min_vocal_ratio=0.5):
        self.min_vocal_ratio = min_vocal_ratio
        if not pre_load:
            self.path = Path(path)
            self.melody_txt_list = [song_id_to_pitch_txt_path(self.path, x) for x in song_ids]
            self.melody_loader = MelodyLoader(in_midi_pitch=True, is_quantized=quantized)
            self.contours = self.load_melody()
        else:
            self.contours = path
        self.num_neg_samples = num_neg_samples
        self.num_aug_samples = num_aug_samples
        self.aug_keys = ['tempo', 'key', 'std', 'masking', 'pitch_noise', 'fill', 'smoothing', 'absurd_noise']
        # self.aug_types = ['different_tempo', 'different_key']
        self.down_f = 10
        self.set_type = set_type

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
        
        # if self.min_aug < len(self.aug_keys):
        #     aug_samples = [self.melody_augmentor(selected_melody, random.sample(self.aug_keys, random.randint(self.min_aug,len(self.aug_keys)))) for i in range(self.num_aug_samples)]
        # else:
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

        

def load_melody(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [float(x.split(' ')[1][:-2]) for x in lines]


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

def song_id_to_pitch_txt_path(path, song_id):
    # path: pathlib.Path()
    txt_path = path / str(song_id)[:3] / str(song_id)[3:6] / '{}_pitch.txt'.format(song_id)
    # txt_path = path / str(song_id)[:3] / str(song_id)[3:6] / 'pitch_{}.txt'.format(song_id)
    if not txt_path.exists():
        txt_path = path / 'qbh' / f'{song_id}_pitch.txt'
    return txt_path
    # return path  / f'pitch_{song_id}.txt'

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
국내 발라드 4 +
해외 힙합 14
기타 30
클래식 20
재즈 22
국내 락/메탈 12 + 
해외 팝 13 + 
월드뮤직 26
해외 락 17 + 
J-POP 25
뉴에이지 21
국내 알앤비 7 + 
CCM 28
해외 알앤비 15 + 
맘/태교 24
OST/BGM 19
국내 댄스/일렉 5
해외 일렉트로닉 16
트로트 8
국내 포크/블루스 11 + 
키즈 23
국내 인디 9 +
종교음악 29
국악 27
국내 팝/어쿠스틱 10 +
국내 힙합 6
'''