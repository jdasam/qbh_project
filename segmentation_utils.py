import numpy as np

def find_melody_seg_fast(contour,zero_threshold, max_length, min_length):
    zeros_slice = get_zero_slice_from_contour(contour, threshold=zero_threshold)
    voice = zero_slice_to_segment(zeros_slice)
    if voice != []:
        expand_voice(voice, max_length=max_length)
    voice = [(int(x[0]), int(x[1])) for x in voice if x[1]-x[0]>min_length]
    return voice


def zero_slice_to_segment(zeros_slice, min_voice_seg=5):
    return [ (zeros_slice[i][1], zeros_slice[i+1][0]) for i in range(len(zeros_slice)-1) if  (zeros_slice[i][1], zeros_slice[i+1][0]) >= min_voice_seg]

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
