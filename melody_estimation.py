import torch
import numpy as np

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