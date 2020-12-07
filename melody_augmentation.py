import random
import numpy as np
import copy
from sampling_utils import downsample_contour, downsample_with_float, downsample_contour_array
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

def with_different_std(x, weight=1):
    aug_x = np.copy(x)
    aug_x[aug_x[:,1]==0,0] = np.nan
    mean = np.nanmean(aug_x[:,0])
    aug_std = (1 + (random.random() - 0.5) *weight)
    is_vocal = x[:,1]==1
    aug_x[is_vocal, 0] -= mean
    aug_x[is_vocal, 0] *= aug_std
    aug_x[is_vocal, 0] += mean
    aug_x[x[:,1]==0] = 0
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

def with_dropout(x, ratio=0.3):
    masking_idx = int(x.shape[0] * random.random())
    masking_len = int(x.shape[0] * ratio)
    out = np.copy(x)
    out[masking_idx:masking_idx + masking_len] = 0
    return out

def with_masking(x, ratio=0.2):
    masking_len = int(x.shape[0] * ratio)
    if masking_len == 0:
        return x
    rand = random.random()
    former_mask_len = int(rand * masking_len)
    later_mask_len = masking_len - former_mask_len

    return x[former_mask_len:-later_mask_len]

    # if rand < 1/3:
    #     return x[masking_len:]
    # elif rand < 2/3:
    #     return x[masking_len//2:-masking_len//2]
    # else:
    #     return x[:-masking_len]
    
def melody_dict_to_array(melody_dict):
    contour = melody_dict['melody']
    is_vocal = melody_dict['is_vocal']
    array = np.stack([contour, is_vocal], axis=-1)
    array[array[:,1]==0, 0] = np.nan
    return np.stack([contour, is_vocal], axis=-1)

def slice_in_random_position(melody_len, max_slice, noise_ratio=4):
    num_slice = random.randint(2, max_slice)
    slice_size = melody_len // num_slice
    noise_size = slice_size // noise_ratio
    approx_slice_position = [int(x) for x in np.linspace(0, melody_len, num_slice)]
    return [approx_slice_position[i] if i in(0, num_slice-1) else approx_slice_position[i] + int((random.random()-0.5) * noise_size) for i in range(num_slice)]

def add_pitch_noise_by_slice(melody, slice_idx, max_noise=0.1):
    dummy = np.copy(melody)
    for i in range(len(slice_idx)-1):
        noise = random.random() * max_noise
        dummy[slice_idx[i]:slice_idx[i+1], 0] = melody[slice_idx[i]:slice_idx[i+1], 0] + noise
    return dummy

def fill_non_voice_entire(melody):
    filled_melody = np.copy(melody)
    filled_melody[:,1] = 1
    if melody[0,1] == 0:
        filled_melody[0,0] = np.nonzero(melody[:,0])[0][0]
        filled_melody[0,1] = 1
    for i in range(1, filled_melody.shape[0]):
        if melody[i, 1] == 0:
            filled_melody[i,0] = filled_melody[i-1,0]
    return filled_melody

def get_zero_slice_from_contour(contour, threshold=5):
    contour_array = np.asarray(contour)
    is_zero_position = np.where(contour_array == 0)[0]
    diff_by_position = np.diff(is_zero_position)
    slice_pos = np.where(diff_by_position>1)[0]
    voice_frame = np.stack([is_zero_position[slice_pos]+1, is_zero_position[slice_pos] + diff_by_position[slice_pos]], axis=-1)
    if voice_frame.shape[0] == 0:
        zeros_slice = []
    else:
        zeros_slice = [ [is_zero_position[0], voice_frame[0,0]] ] + [ [voice_frame[i-1,1], voice_frame[i,0]] for i in range(1, voice_frame.shape[0])]
        zeros_slice = [x for x in zeros_slice if x[1]-x[0] > threshold]
    return zeros_slice

def fill_non_voice_random(melody, max_ratio=1, pause_threshold=200):
    slice_pos = get_zero_slice_from_contour(melody[:,1])
    long_pause_idx = [i for i in range(len(slice_pos)) if slice_pos[i][1]-slice_pos[i][0] > pause_threshold]
    for i in reversed(long_pause_idx):
        deleted = slice_pos.pop(i)
#         num_non_voice -= deleted[1] - deleted[0]
    num_non_voice = sum([x[1]-x[0] for x in slice_pos])
    num_fill_voice = random.random() * max_ratio * num_non_voice
    filled_melody = np.copy(melody)
    filled = 0
    while filled < num_fill_voice:
        sl = slice_pos.pop(random.randint(0, len(slice_pos)-1))
        filled_melody[sl[0]:sl[1], 0] = melody[sl[0]-1, 0]
        filled_melody[sl[0]:sl[1], 1] = 1
        filled += sl[1] - sl[0]
    return filled_melody


def downsample_with_different_tempo(melody, global_tempo, slice_ids, rand_weight=0.2):
    downsampled = [downsample_with_float(melody[slice_ids[i]:slice_ids[i+1]], down_f=10*(global_tempo+random.random()*rand_weight)) for i in range(len(slice_ids)-1) ]
    return np.concatenate(downsampled)


def make_augmented_melody(melody_array, aug_keys):
    # 1. masking in random ratio
    if 'masking' in aug_keys:
        masking_ratio = random.random() / 2
        aug_melody = with_masking(melody_array, ratio=masking_ratio)
        while np.sum(aug_melody[:,1]) == 0:
            aug_melody = with_masking(melody_array, ratio=masking_ratio)
    else:
        aug_melody = np.copy(melody_array)
    
    if 'tempo' in aug_keys:
        global_tempo = random.random() + 0.5
        slice_ids = slice_in_random_position(aug_melody.shape[0], 7)
        aug_melody = downsample_with_different_tempo(aug_melody, global_tempo, slice_ids)
        while np.sum(aug_melody[:,1]) == 0:
            if 'masking' in aug_keys:
                masking_ratio = random.random() / 2
                aug_melody = with_masking(melody_array, ratio=masking_ratio)
                while np.sum(aug_melody[:,1]) == 0:
                    aug_melody = with_masking(melody_array, ratio=masking_ratio)
            else:
                aug_melody = np.copy(melody_array)
            global_tempo = random.random() + 0.5
            slice_ids = slice_in_random_position(aug_melody.shape[0], 7)
            aug_melody = downsample_with_different_tempo(aug_melody, global_tempo, slice_ids)
    else:
        aug_melody = downsample_contour_array(aug_melody)
        
    if 'drop_out' in aug_keys:
        aug_melody = with_dropout(aug_melody)

    if 'key' in aug_keys:
        aug_melody = with_different_key(aug_melody)
    
    if 'std' in aug_keys:
        aug_melody = with_different_std(aug_melody, weight=0.3)

    if 'pitch_noise' in aug_keys:
        # 4. add different pitch noise by slice
        slice_ids = slice_in_random_position(aug_melody.shape[0], 5)
        aug_melody = add_pitch_noise_by_slice(aug_melody, slice_ids)

    if 'fill' in aug_keys:
        # randomly fill non voice part 
        aug_melody = fill_non_voice_random(aug_melody)
    aug_melody[aug_melody[:,1]==0, 0] = 0
    return aug_melody