from utils.data_path_utils import song_id_to_pitch_txt_path, song_id_to_audio_path
import numpy as np
from pathlib import Path
from utils.melody_extraction_utils import MelodyExtractor
MEAN = 61.702336487738215
STD = 5.5201786930065415

class MelodyLoader:
    def __init__(self, dir_path, win_size=2000, hop_size=500, min_ratio=0.5, device='cuda'):
        self.data_dir = Path(dir_path)
        self.win_size = win_size
        self.hop_size = hop_size
        self.min_ratio = min_ratio

        self.melody_extractor = MelodyExtractor(device=device)

    def check_txt_exists(self, song_ids, make_txt=False, verbose=False):
        everything_ok = True
        for idx in song_ids:
            path = song_id_to_pitch_txt_path(self.data_dir, idx)
            if not path.exists():
                if verbose:
                    print(f"Melody txt for Song ID {idx} does not exist")
                if make_txt:
                    audio_path = song_id_to_audio_path(self.data_dir, idx)
                    if audio_path.exists():
                        self.melody_extractor(audio_path)
                    else:
                        if verbose:
                            print(f"Audio file for Song ID {idx} does not exist")
                        everything_ok = False
                else:
                        everything_ok = False
        if everything_ok:
            print("Check passed: Every song has corresponding txt")

    def __call__(self, song_id):
        path = song_id_to_pitch_txt_path(self.data_dir, song_id)
        return get_overlapped_contours(path, self.win_size, self.hop_size, self.min_ratio)

def get_overlapped_contours(path, win_size=2000, hop_size=500, min_ratio=0.5):
    contour = load_melody(path)
    melody_form = scale_to_midi(np.asarray(contour))
    melody_form = pitch_array_to_formatted(melody_form)
    # melody_ds = downsample_contour(contour)
    slice_pos = list(range(0, melody_form.shape[0] - win_size, hop_size))
    slice_pos.append(melody_form.shape[0] - win_size)
    array_overlapped = np.asarray([melody_form[i:i+win_size] for i in slice_pos])
    is_valid = np.sum(array_overlapped[:,:,1], axis=1) > win_size * min_ratio
    overlapped_melodies = [{'contour': melody_form[i:i+win_size],
                            # 'song_id': int(path.stem[6:]),
                            'song_id': int(path.stem[:-6]),
                            'frame_pos': (i, i+win_size)} for n, i in enumerate(slice_pos)
                            # if sum(melody_form[i:i+win_size, 1]) > win_size * min_ratio]
                            if is_valid[n]]
    return overlapped_melodies



def scale_to_midi(contour):
    # contour[contour[:,1]==1,0] = np.log2(contour[contour[:,1]==1,0] / 440) * 12 + 69
    is_pitch = np.nonzero(contour)
    contour_array = np.copy(contour)
    contour_array[is_pitch] = np.log2(contour_array[is_pitch] / 440) * 12 + 69
    return contour_array

def load_melody(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [float(x.split(' ')[1][:-2]) for x in lines]
    # return np.loadtxt(path, dtype=np.float32, delimiter=' ')[:,1]


def pitch_array_to_formatted(pitch_array, mean=MEAN, std=STD):
    output = np.zeros((len(pitch_array), 2))
    output[pitch_array!=0,1] = 1
    output[:,0] = (pitch_array - mean) / std
    output[output[:,1]==0, 0]= 0
    return output

def melody_to_formatted_array(melody):
    melody_in_midi = scale_to_midi(np.asarray(melody))
    return pitch_array_to_formatted(melody_in_midi)

# if __name__ == '__main__':
#     dataset = MelodyDataset('/home/svcapp/userdata/musicai/flo_data/')
#     dataset.save('/home/svcapp/userdata/flo_melody/melody_entire.dat')
#     # pitch_path = '/home/svcapp/userdata/musicai/dev/teo/melodyExtraction_JDC/output/pitch_435845929.txt'
#     # loader = MelodyLoader()
#     # tokens = loader(pitch_path)

