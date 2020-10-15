import copy
import numpy as np

def downsample_contour(contour, is_vocal=None, down_f=10, down_type='sample'):
    contour = copy.copy(contour)
    if len(contour) // down_f != 0:
        num_pad = down_f - len(contour) % down_f
        contour += [contour[-1]] * num_pad
        if isinstance(is_vocal, list):
            is_vocal = copy.copy(is_vocal)
            is_vocal += [is_vocal[-1]] * num_pad
        # for i in range(down_f - len(contour) % down_f):
        #     contour.append(contour[-1])
    contour_array = np.asarray(contour, dtype=float)
    contour_d = contour_array.reshape(-1,down_f)
    if down_type == 'mean':
        contour_d = np.nanmean(contour_d, axis=1)
        contour_d[np.isnan(contour_d)] = 0
    elif down_type == 'median':
        contour_d = np.nanmedian(contour_d, axis=1)
        
        contour_d[np.isnan(contour_d)] = 0
    else:
        contour_d  = contour_d[:, 0]

    if isinstance(is_vocal, list):
        is_vocal_array = np.asarray(is_vocal, dtype=float)
        is_vocal_d = is_vocal_array.reshape(-1,down_f)
        if down_type == 'mean':
            is_vocal_d = np.nanmean(is_vocal_d, axis=1)
            is_vocal_d[np.isnan(is_vocal_d)] = 0
        elif down_type == 'median':
            is_vocal_d = np.nanmedian(contour_d, axis=1)
            is_vocal_d[np.isnan(is_vocal_d)] = 0
        else:
            is_vocal_d  = is_vocal_d[:, 0]
        return np.stack([contour_d, is_vocal_d], axis=-1)
    else:
        return contour_d
