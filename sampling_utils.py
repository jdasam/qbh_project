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


def downsample_with_float_list(contour, is_vocal, down_f=10, down_type='median'):
    # input: list of pitch, list of is_vocal
    # output: numpy array with L X 2. downsampled
    ds_slice_idx = [int(x*down_f) for x in range(int(len(contour)//down_f)+1)]
    end_slice_idx = ds_slice_idx[1:] + [len(contour)]
    input_array = np.stack([contour, is_vocal]).T
    input_array[input_array[:,1]==0,0] = np.nan
    ds_contour = [np.nanmedian(np.asarray(contour[x:y]))for x,y in zip(ds_slice_idx, end_slice_idx)]
    ds_is_vocal = [np.nanmedian(np.asarray(is_vocal[x:y]))for x,y in zip(ds_slice_idx, end_slice_idx)]
    ds_contour[ds_is_vocal==0] = 0
    ds_contour = np.stack([ds_contour, ds_is_vocal]).T

    return ds_contour


def downsample_with_float(contour_array, down_f=10, down_type='sample'):
    # input: array of 
    # output: numpy array with L X 2. downsampled
    contour_array = np.copy(contour_array)
    if down_type=='sample':
        ds_slice_idx = [int(x*down_f) for x in range(int(len(contour_array)//down_f))]
        contour_array[contour_array[:,1]==0,0] = np.nan
        ds_contour = np.stack([contour_array[x,0] for x in ds_slice_idx])
        ds_is_vocal = np.stack([contour_array[x,1] for x in ds_slice_idx])
    else:
        ds_slice_idx = [int(x*down_f) for x in range(int(len(contour_array)//down_f)+1)]
        end_slice_idx = ds_slice_idx[1:] + [len(contour_array)]
#     ds_contour = np.stack([np.nanmedian(contour_array[x:y,0]) for x,y in zip(ds_slice_idx, end_slice_idx)])
        ds_contour = np.stack([np.nanmedian(contour_array[x:y,0]) if np.sum(contour_array[x:y,1])>0 else 0 for x,y in zip(ds_slice_idx, end_slice_idx)])
        ds_is_vocal = np.stack([np.sum(contour_array[x:y,1])> (y-x)//3 for x,y in zip(ds_slice_idx, end_slice_idx)])
    ds_contour[ds_is_vocal==0] = 0
    ds_contour = np.stack([ds_contour, ds_is_vocal]).T

    return ds_contour