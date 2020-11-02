from math import log2
import torch
from pathlib import Path
import _pickle as pickle

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
    def __init__(self, path, pre_load=False, min_vocal_len=20):
        self.path = Path(path)
        self.melody_loader = MelodyLoader()

        self.pitch_list = list(self.path.rglob('*.txt'))
        print('Number of total pitch txt: {}'.format(len(self.pitch_list)))
        self.pre_load = pre_load
        if self.pre_load:
            self.load_lists_of_melody_txt()
        # self.non_vocal_index = []
        self.min_vocal_len = min_vocal_len

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
        datas_in_token = [self.melody_loader(txt) for txt in self.pitch_list]
        self.datas_in_token = [x for x in datas_in_token if cal_total_voice_sec(x)> self.min_vocal_len]
        
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
    def __init__(self):
       self.q_pitch, self.q_boundary = make_quantization_info(low_pitch=110)
       self.patience = 100
       self.min_mel_length = 100

    def melody_quantize(self, pitch_contour, idx):
        melodies = []
        continuing = False
        melody = []
        pitch_notes = []
        pitch_event = []
        non_pitch_count = 0
        for i in range(len(pitch_contour)):
            pitch = pitch_contour[i]
            if pitch != 0:
                # melody.append(pitch)
                q_pitch = self.q_pitch[binary_index(self.q_boundary, pitch)+1]
                melody.append(q_pitch)
                non_pitch_count = 0
                continuing = True
                if len(pitch_event) == 0 or pitch_event[-1] == q_pitch:
                    pitch_event.append(q_pitch)
                else:
                    # if len(melody) > min_pitch_length:
                    pitch_notes.append(pitch_event)
                    pitch_event=[q_pitch]
            else: # if pitch is zero
                non_pitch_count += 1
                if non_pitch_count > self.patience:  # too many zeros are continuing, so let's say the melody ended
                    if continuing:
                        if len(melody) > self.min_mel_length:
                            melodies.append({'melody': melody, 'frame_pos': (i-len(melody), i), 'idx': idx})
                            pitch_notes.append(pitch_event)
                        melody = []
                        pitch_event=[]
                        continuing = False
                        non_pitch_count = 0
                else:
                    if continuing:
                        melody.append(melody[-1])
                        # melody.append(pitch) # append zero to melody
                        pitch_event.append(pitch_event[-1]) # duplicate the end of pitch_event
        return melodies
    
    def __call__(self, path):
        pitch_contour = load_melody(path)
        quantized_melodies = self.melody_quantize(pitch_contour, idx=path.stem)
        melody_in_token = [melody_to_token(melody) for melody in quantized_melodies]
        melody_in_token = [x for x in melody_in_token if x['tokens'] != []]
        return melody_in_token

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

def hz_to_midi_pitch(hz, quantization=True):
    if hz == 0:
        return 0
    if quantization:
        return round(log2(hz/440) * 12) + 69
    else:
        return log2(hz/440) * 12 + 69


def normalize_contour_with_stat(contour, mean=62.28434563742319, std=5.645042479045596):
    return [normalize_value_if_not_zero(x, mean, std) for x in contour]

def normalize_value_if_not_zero(value, mean, std):
    if value == 0:
        return value
    else:
        return (value-mean) / std

def cleaning_query_input(frequency, hop_size=10):
    for i in range(0, len(frequency)-hop_size, hop_size):
        windowed = frequency[i:i+hop_size]
        zero_count = windowed.count(0)
        if zero_count < 4:
            break
    cleaned_frequency = frequency[i:]
    for i in range(len(cleaned_frequency)-hop_size, 0, -hop_size):
        windowed = frequency[i:i+hop_size]
        zero_count = windowed.count(0)
        if zero_count < 5:
            break
    cleaned_frequency = cleaned_frequency[:i+hop_size]
    return cleaned_frequency

def melody_to_token(melody, min_pitch_length=10):
    tokens = {'tokens':[], 'frame_pos':melody['frame_pos'], 'idx':melody['idx']}
    prev_pitch = melody['melody'][0]
    pitch_duration = 0
    for pitch in melody['melody']:
        if pitch == prev_pitch:
            pitch_duration += 1
        else:
            if pitch_duration > min_pitch_length:
                tokens['tokens'].append({'pitch': hz_to_midi_pitch(prev_pitch), 'duration': pitch_duration})
            pitch_duration = 1
            prev_pitch = pitch
    return tokens

def token_to_list(in_seq):
    # [ (in_seq[i][0] - in_seq[i], in_seq[i][1]) for i in range(1, len(in_seq))]
    # return [ (note['pitch'] - 44, duration_to_class(note['duration'])) for note in in_seq ]
    seq = [ (note['pitch'], duration_to_class(note['duration'])) for note in in_seq ]
    init = [(50, seq[0][1])]
    init += [(seq[i][0]- seq[i-1][0]  + 50, seq[i][1]) for i in range(1,len(seq))]
    return init


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


if __name__ == '__main__':
    dataset = MelodyDataset('/home/svcapp/userdata/musicai/flo_data/')
    dataset.save('/home/svcapp/userdata/flo_melody/melody_entire.dat')
    # pitch_path = '/home/svcapp/userdata/musicai/dev/teo/melodyExtraction_JDC/output/pitch_435845929.txt'
    # loader = MelodyLoader()
    # tokens = loader(pitch_path)

