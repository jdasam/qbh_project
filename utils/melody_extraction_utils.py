'''
This module is a modified version of https://github.com/keums/melodyExtraction_SSL, 
an implementation of "Semi-supervised learning using teacher-student models for vocal melody extraction", ISMIR(2020) by Sangeun Kum.
Please refer https://arxiv.org/abs/2008.06358 for the detail
'''

import torch
import numpy as np
from pathlib import Path
import librosa
from madmom.audio.signal import *
from model.model import Melody_ResNet

class MelodyExtractor:
    def __init__(self, weight_dir=Path('weights/'), device='cuda'):
        self.model = Melody_ResNet()
        self.model.load_state_dict(torch.load(weight_dir / 'melody_extractor_weights.pt'))
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device

        self.spec_mean = np.load(weight_dir / 'x_data_mean_total_31.npy')
        self.spec_std = np.load(weight_dir / 'x_data_std_total_31.npy')

        note_res = 8
        pitch_range = np.arange(40, 95 + 1.0/note_res, 1.0/note_res)
        self.pitch_range = np.concatenate([np.zeros(1), pitch_range])

    def __call__(self, file_name):
        file_name = Path(file_name)
        X_test, _ = self.spec_extraction(file_name=str(file_name), win_size=31)
        
        '''  melody predict'''
        y_predict = self.run_neural_net(X_test)
        est_pitch = self.convert_output_to_melody(y_predict)
        save_pitch_estimation_in_txt(file_name, est_pitch)

    def run_neural_net(self, input_spec):
        with torch.no_grad():
            self.model.eval()
            torch_input = torch.Tensor(input_spec).permute(0,3,1,2)
            torch_input = torch_input.to(self.device)
            y_predict = self.model.batch_slice_and_forward(torch_input)
            y_predict = y_predict.cpu().numpy()
        return y_predict

    def convert_output_to_melody(self, y_predict):
        y_shape = y_predict.shape
        num_total_frame = y_shape[0]*y_shape[1]
        est_pitch = np.zeros(num_total_frame)
        index_predict = np.zeros(num_total_frame)

        y_predict = np.reshape(y_predict, (num_total_frame, y_shape[2]))

        for i in range(num_total_frame):
            index_predict[i] = np.argmax(y_predict[i, :])
            pitch_MIDI = self.pitch_range[np.int32(index_predict[i])]
            if pitch_MIDI >= 45 and pitch_MIDI <= 95:
                est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440
        return est_pitch

    def load_audio(self, file_name):
        return Signal(file_name, sample_rate=8000, dtype=np.float32, num_channels=1)

    def get_melody_from_audio(self, audio_sample):
        x_test, _ = self.get_norm_spec_from_audio(audio_sample)
        y_predict = self.run_neural_net(x_test)
        est_pitch = self.convert_output_to_melody(y_predict)
        return est_pitch


    def get_norm_spec_from_audio(self, audio_sample, win_size=31):
        x_test = []
        S = librosa.core.stft(audio_sample, n_fft=1024, hop_length=80*1, win_length=1024)
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

        # for normalization

        x_test = (x_test-self.spec_mean)/(self.spec_std+0.0001)
        x_test = x_test[:, :, :, np.newaxis]

        return x_test, x_spec

    def spec_extraction(self, file_name):
        # y, sr = librosa.load(file_name, sr=8000)
        # *********** madmom.Signal() is faster than librosa.load() ***********
        y = self.load_audio(file_name)
        x_test, x_spec = self.get_norm_spec_from_audio(y)

        return x_test, x_spec

    

def save_pitch_estimation_in_txt(file_name:Path(), est_pitch):
    PATH_est_pitch = file_name.parent / (file_name.stem + '_pitch.txt')
    f = open(PATH_est_pitch, 'w')
    for j in range(len(est_pitch)):
        est = "%.2f %.4f\n" % (0.01 * j, est_pitch[j])
        f.write(est)
    f.close()



global pitch_hz, pitch_range

note_res = 8
pitch_range = np.arange(40, 95 + 1.0/note_res, 1.0/note_res)
pitch_range = np.concatenate([np.zeros(1), pitch_range])
pitch_hz = 2** ((pitch_range-69) / 12) * 440
pitch_hz[0] = 0

def model_prediction_to_pitch(pred, to_hz=False):
    pitch_class= np.argmax(pred, axis=-1)
    if to_hz:
        pitch_table = pitch_hz
    else:
        pitch_table = pitch_range
    return pitch_range[pitch_class]

def elongate_result(pred, ratio=10):
    return np.repeat(pred, ratio, axis=-1)