from math import log2
import torch
from pathlib import Path

import _pickle as pickle
from tqdm.auto import tqdm
import numpy as np
import random
import copy
import melody_augmentation as mel_aug
from sampling_utils import downsample_contour_array
from melody_utils import get_overlapped_contours
from data_path_utils import song_id_to_pitch_txt_path, song_id_to_audio_path
from madmom.audio.signal import Signal


class WindowedContourSet:
    def __init__(self, path, aug_weights, song_ids=[], num_aug_samples=4, num_neg_samples=4, pre_load_data=None, set_type='entire', min_vocal_ratio=0.5):
        self.min_vocal_ratio = min_vocal_ratio
        self.path = Path(path)
        if pre_load_data is None:
            self.melody_txt_list = [song_id_to_pitch_txt_path(self.path, x) for x in song_ids]
            # self.melody_loader = MelodyLoader(self.path, min_ratio=min_vocal_ratio)
            self.contours = self.load_melody()
        else:
            self.contours = pre_load_data
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
        contours = [get_overlapped_contours(txt, min_ratio=self.min_vocal_ratio) for txt in tqdm(self.melody_txt_list, leave=False)]
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



class AudioSet(WindowedContourSet):
    def __init__(self, path, aug_weights, song_ids=[], num_aug_samples=4, num_neg_samples=4, pre_load_data=None, set_type='entire', min_vocal_ratio=0.5, sample_rate=8000):
        # super(AudioSet, self).__init__(path, aug_weights, song_ids, num_aug_samples, num_neg_samples, pre_load, set_type, min_vocal_ratio)    
        super().__init__(path, aug_weights, song_ids, num_aug_samples, num_neg_samples, pre_load_data, set_type, min_vocal_ratio)    
        self.sample_rate = sample_rate
        # self.x_train_mean = np.load('x_data_mean_total_31.npy')
        # self.x_train_std = np.load('x_data_std_total_31.npy')

    def load_audio(self, song_id, frame_pos):
        audio_path = song_id_to_audio_path(self.path, song_id)
        audio_samples = load_audio_sample(audio_path, self.sample_rate)
        # x_test, x_spec = load_audio_and_get_spec(audio_path, frame_pos)
        # x_test = (x_test-self.x_train_mean)/(self.x_train_std+0.0001)
        # x_test = x_test[:, :, :, np.newaxis]
        return audio_samples[frame_pos[0]//100*self.sample_rate:frame_pos[1]//100*self.sample_rate]
    

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
        downsampled_melody = downsample_contour_array(selected_melody)
        if self.set_type != "valid":
            original_audio = self.load_audio(selected_song_id, selected_frame)

        if self.set_type == 'entire':
            return original_audio, selected_song_id

        aug_samples = []
        neg_samples = []
        # neg_audio_samples = []
        
        aug_samples = [self.melody_augmentor(selected_melody, self.aug_keys) for i in range(self.num_aug_samples-1)]
        aug_samples = [downsampled_melody] + aug_samples
        if self.set_type == 'valid':
            return aug_samples, [selected_song_id] * len(aug_samples)
            # return [downsampled_melody] * len(aug_samples), [selected_song_id] * len(aug_samples)

        # sampling negative melodies
        while len(neg_samples) < self.num_neg_samples:
            neg_idx = random.randint(0, len(self)-1)
            if self.contours[neg_idx]['song_id'] != selected_song_id:
                neg_samples.append(downsample_contour_array(self.contours[neg_idx]['contour'], self.down_f))
                # neg_audio_samples.append(self.load_audio(self.contours[neg_idx]['song_id'], self.contours[neg_idx]['frame_pos']))
                # neg_samples.append(self.contours[neg_idx]['song_id'])

        # return original_audio, neg_audio_samples, aug_samples, neg_samples
        return original_audio, aug_samples, neg_samples


class HummingAudioSet(HummingPairSet):
    def __init__(self, path, contour_pairs, aug_weights, set_type, aug_keys, num_aug_samples=4, num_neg_samples=4, sample_rate=8000):
        super(HummingAudioSet, self).__init__(contour_pairs, aug_weights, set_type, aug_keys, num_aug_samples=num_aug_samples, num_neg_samples=num_neg_samples)
        self.sample_rate = sample_rate
        self.path = Path(path)


    def load_audio(self, song_id, time_stamp):
        audio_path = song_id_to_audio_path(self.path, song_id)
        audio_samples = load_audio_sample(audio_path, self.sample_rate)
        return audio_samples[time_stamp[0]*self.sample_rate:time_stamp[1]*self.sample_rate]
    
    def __getitem__(self, index):
        """
        for training:
        return: (downsampled_melody, [augmented_melodies], [negative_sampled_melodies])
        for validation:
        return: ([augmented_melodies], [selected_song_id])
        """
        selected_melody = self.contours[index]['humm']
        # original_melody = self.contours[index]['orig']
        selected_song_id = self.contours[index]['meta']['track_id']
        # orig_ds_melody = downsample_contour_array(original_melody)
        downsampled_melody = downsample_contour_array(selected_melody)
        orig_sample = self.load_audio(selected_song_id, [int(x) for x in self.contours[index]['meta']['time_stamp'].split('-')])

        aug_samples = []
        neg_samples = []
        # neg_audio_samples = []
        
        if self.set_type == 'valid' or self.set_type == 'test':
            return downsampled_melody, selected_song_id
            # return [downsampled_melody] * len(aug_samples), [selected_song_id] * len(aug_samples)
        
        aug_samples = [self.melody_augmentor(selected_melody,self.aug_keys) for i in range(self.num_aug_samples-1)]
        aug_samples = [downsampled_melody] + aug_samples

        # sampling negative melodies
        while len(neg_samples) < self.num_neg_samples:
            neg_idx = random.randint(0, len(self)-1)
            if self.contours[neg_idx]['meta']['track_id'] != selected_song_id:
                neg_samples.append(downsample_contour_array(self.contours[neg_idx]['humm'], self.down_f))
                # neg_audio_samples.append(self.load_audio(self.contours[neg_idx]['meta']['track_id'], [int(x) for x in self.contours[neg_idx]['meta']['time_stamp'].split('-')]))

        return orig_sample, aug_samples, neg_samples



class AudioContourCollate:
    # def __init__(self, num_pos, num_neg, for_cnn=False):
    #     self.num_pos = num_pos
    #     self.num_neg = num_neg

    def to_tensor_list(self, alist):
        return [torch.Tensor(x) for x in alist]

    def make_tensor_with_auto_pad(self, alist):
        max_length = max([len(x) for x in alist])
        dummy = torch.zeros(len(alist), max_length)
        for i in range(len(alist)):
            seq = alist[i]
            left_margin = (max_length - seq.shape[0]) // 2
            dummy[i,left_margin:left_margin+seq.shape[0]] = seq
        return dummy


    def __call__(self, batch):
        # batch: [(audio_anchor, positive_contour, negative_contour) ]* num_batch
        # anchor_audio = torch.Tensor([x[0] for x in batch])
        anchor_audio = [torch.Tensor(np.copy(x[0])) for x in batch]
        anchor_audio = self.make_tensor_with_auto_pad(anchor_audio)
        # anchor_and_neg_audio = [ [torch.Tensor(np.copy(x[0]))] + self.to_tensor_list(x[1]) for x in batch]
        # anchor_and_neg_audio = [y for x in anchor_and_neg_audio for y in x]
        # anchor_and_neg_audio = self.make_tensor_with_auto_pad(anchor_and_neg_audio)

        # total = [self.to_tensor_list(x[2]) + self.to_tensor_list(x[3]) for x in batch ]
        total = [self.to_tensor_list(x[1]) + self.to_tensor_list(x[2]) for x in batch ]
        total_flattened = [y for x in total for y in x]
        max_length = max([len(x) for x in total_flattened])
        dummy = torch.zeros(len(total_flattened), max_length, 2)
        for i in range(len(total_flattened)):
            seq = total_flattened[i]
            left_margin = (max_length - seq.shape[0]) // 2
            dummy[i,left_margin:left_margin+seq.shape[0],:] = seq
        # return anchor_and_neg_audio, dummy
        return anchor_audio, dummy

class AudioCollate:
    def __call__(self, batch):
        out = [torch.Tensor(x[0]) for x in batch]
        song_ids = torch.LongTensor([x[1] for x in batch])

        max_length = max([len(x) for x in out])
        dummy = torch.zeros(len(out), max_length)
        for i in range(len(out)):
            seq = out[i]
            left_margin = (max_length - seq.shape[0]) // 2
            dummy[i,left_margin:left_margin+seq.shape[0]] = seq
        return dummy, song_ids



class AudioTestSet:
    def __init__(self, path, song_ids=[], sample_rate=8000):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.song_ids = song_ids

        self.slice_infos = [self.cal_slice_position(x) for x in self.song_ids]
        self.slice_infos = [y for x in self.slice_infos for y in x]

    def cal_slice_position(self, song_id):
        audio_path = song_id_to_audio_path(self.path, song_id)
        audio_samples = load_audio_sample(audio_path, self.sample_rate)
        num_window = (len(audio_samples) - self.sample_rate * 20) // (self.sample_rate * 5) + 1
        return [(song_id, i * self.sample_rate * 5, i * self.sample_rate * 5 + 20 * self.sample_rate) for i in range(num_window)]

    # def cal_num_window(self, song_id):
    #     audio_path = song_id_to_audio_path(self.path, song_id)
    #     audio_samples = load_audio_sample(audio_path, self.sample_rate)
    #     return (len(audio_samples) - self.sample_rate * 20) // (self.sample_rate * 5) + 1

    def load_audio(self, song_id, start, end):
        audio_path = song_id_to_audio_path(self.path, song_id)
        audio_samples = load_audio_sample(audio_path, self.sample_rate)
        return audio_samples[start:end]

    def __getitem__(self, index):
        sel_id, sel_start, sel_end = self.slice_infos[index]
        audio = self.load_audio(sel_id, sel_start, sel_end)
        return audio, sel_id

    def __len__(self):
        return len(self.slice_infos)
    

def load_melody(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [float(x.split(' ')[1][:-2]) for x in lines]


def load_audio_sample(path, sr=8000):
    '''
    For faster loading, the audio samples are decoded and saved in npy file
    '''
    npy_path = path.with_suffix('.npy')
    if npy_path.exists():
        try:
            y = np.load(npy_path)
        except:
            print(f"error occurned on {npy_path}")
            y = Signal(str(path), sample_rate=sr, dtype=np.float32, num_channels=1)
            np.save(npy_path, y, allow_pickle=False)
    else:
        y = Signal(str(path), sample_rate=sr, dtype=np.float32, num_channels=1)
        np.save(npy_path, y, allow_pickle=False)
    
    # Code below is to prevent y from being saved in spectrogram form. 
    # Couldn't find the reason why y is saved in spectrogram
    if len(y)==128:
        y = Signal(str(path), sample_rate=sr, dtype=np.float32, num_channels=1)
        np.save(npy_path, y, allow_pickle=False)
        print(path, npy_path)
    return np.copy(y)

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