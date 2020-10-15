import random
import numpy as np
import copy
from sampling_utils import downsample_contour
# Input x is an np.array of pitch of each frame

aug_types = ['different_tempo', 'different_key', 'different_std', 'addition', 'masking']
tempo_down_f = [5,6,7,8,9,11,12,13,14,15,16]


def with_different_tempo(x, is_vocal):
    down_f = random.sample(tempo_down_f, 1)
    return downsample_contour(x, is_vocal, down_f=down_f[0])

def with_different_key(x):
    aug_x = np.copy(x)
    is_vocal = aug_x[:,1]==1
    aug_x[is_vocal,0] += (random.random()-0.5) * 3
    return aug_x

def with_different_std(x):
    mean = np.mean(x[:,0])
    std = np.std(x[:,0])
    aug_std = std * (random.random() + 0.5)
    is_vocal = x[:,1]==1
    aug_x = np.copy(x)
    aug_x[is_vocal, 0] -= mean
    aug_x[is_vocal, 0] *= aug_std
    aug_x[is_vocal, 0] += mean
    return aug_x
    # return np.stack([(x[:,0]-mean) * aug_std + mean, x[:,1]], axis=-1) 

def with_addition(x, ratio=0.2):
    old_len = x.shape[0]
    new_len = int(x.shape[0] * (1+ratio))
    aug_x = np.zeros((new_len, 2))
    aug_index = random.sample(range(old_len), new_len-old_len)
    aug_index += list(range(old_len))
    aug_index.sort()
    
    aug_x[list(range(new_len))] = x[aug_index]
    return aug_x

def with_reduction(x, ratio=0.2):
    return


def with_masking(x, ratio=0.2):
    masking_len = int(x.shape[0] * ratio)
    rand = random.random()
    if rand < 1/3:
        return x[masking_len:]
    elif rand < 2/3:
        return x[masking_len//2:-masking_len//2]
    else:
        return x[:-masking_len]