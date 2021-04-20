import numpy as np
import shutil
import soundfile
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pathlib import Path

from data_utils import song_id_to_pitch_txt_path
from humming_data_utils import load_crepe_pitch as load_humm_melody, get_orig_audio_path_by_id, audio_path_to_pitch_path
from melody_utils import pitch_array_to_formatted, load_melody, scale_to_midi
from sampling_utils import downsample_contour_array

def generate_sine_wav(melody, frame_rate=10, sr=44100):
    melody_resampled = np.repeat(melody, sr//frame_rate)
    phi = np.zeros_like(melody_resampled)
    phi[1:] = np.cumsum(2* np.pi * melody_resampled[:-1] / sr, axis=0)
    sin_wav = 0.9 * np.sin(phi)
    return sin_wav

def normalized_vec_to_orig(norm_contour, mean_pitch=61.702336487738215, std=5.5201786930065415):
    orig = np.zeros_like(norm_contour[:,0])
    orig[norm_contour[:,1]==1] = 440 * 2 ** ((norm_contour[norm_contour[:,1]==1, 0] * std + mean_pitch -69) / 12)
#     orig[norm_contour==-100] = 0
    return orig

def plot_contour_with_voice_only(contour):
    dummy = np.copy(contour)
    dummy[contour[:,1]==0,0] = np.nan
    plt.plot(dummy[:,0])

def id_to_name(idx, meta):
    if 'artist_name' in meta[idx]:
        name = f'{meta[idx]["artist_name"]} - {meta[idx]["track_name"]}'
    else:
        name = f'{meta[idx]["artist_name_basket"][0]} - {meta[idx]["track_name"]}'
    return name.replace("/", "_") 

def save_test_result_in_wav(total_recommends, total_test_ids, total_rank, total_rec_slices, meta, humm_meta, out_dir, db_path=Path('/home/svcapp/t2meta/flo_new_music/music_100k/')):
    for i in range(len(humm_meta)):
        track_id = humm_meta[i]["track_id"]
        track_name = id_to_name(track_id, meta)
        out_path = str(out_dir / f"{i}_rankResult_{total_rank[i]}_{track_name}")

        #1 copy humming.wav
        shutil.copy(humm_meta[i]['path'], out_path+"_humm.wav")

        #2 save humming contour in wav
        humm_contour = load_humm_melody(humm_meta[i]['pitch_path'], to_midi_pitch=True)
        humm_contour_downsampled = downsample_contour_array(pitch_array_to_formatted(humm_contour))
        
        humm_contour_wav = generate_sine_wav(normalized_vec_to_orig(humm_contour_downsampled), frame_rate=10, sr=16000)
        soundfile.write(out_path+"_humm_contour.wav", humm_contour_wav, samplerate=16000)
        
        #3 save first recommends in wav
        rec_id = total_recommends[i,0]
        rec_name = id_to_name(rec_id, meta)
        rec_pitch_path = song_id_to_pitch_txt_path(db_path, rec_id)
        rec_contour = pitch_array_to_formatted(scale_to_midi(np.asarray(load_melody(rec_pitch_path))))
        rec_contour = rec_contour[total_rec_slices[i,0,0]:total_rec_slices[i,0,1]]
        rec_contour_downsampled = downsample_contour_array(rec_contour)
        rec_contour_wav = generate_sine_wav(normalized_vec_to_orig(rec_contour_downsampled), frame_rate=10, sr=16000)
        soundfile.write(out_path+f"_rec1_contour_{rec_name}.wav", rec_contour_wav, samplerate=16000)

        #4 save rec slice in wav
        rec_audio_path = get_orig_audio_path_by_id(rec_id, db_path)
        rec_song = AudioSegment.from_file(rec_audio_path, 'm4a').set_channels(1)._data
        rec_decoded = np.frombuffer(rec_song, dtype=np.int16) / 32768
        soundfile.write(out_path+f"_re1_{rec_name}.wav", rec_decoded[total_rec_slices[i,0,0]*441:total_rec_slices[i,0,1]*441], samplerate=44100)


        #5 save orig slice in wav
        orig_audio_path = get_orig_audio_path_by_id(track_id, db_path)
        orig_song = AudioSegment.from_file(orig_audio_path, 'm4a').set_channels(1)._data
        orig_decoded = np.frombuffer(orig_song, dtype=np.int16) / 32768
        
        time_pos = humm_meta[i]['time_stamp'].split('-')
        start_position = int(time_pos[0]) * 44100
        end_position = int(time_pos[1]) * 44100
        soundfile.write(out_path+f"_orig.wav", orig_decoded[start_position:end_position], samplerate=44100)

        #6 save orig contour in wav
        orig_pitch_path = audio_path_to_pitch_path(orig_audio_path)
        orig_contour = pitch_array_to_formatted(scale_to_midi(np.asarray(load_melody(orig_pitch_path))))
        orig_contour = orig_contour[int(time_pos[0]) * 100: int(time_pos[1]) * 100]
        orig_contour_downsampled = downsample_contour_array(orig_contour)
        orig_contour_wav = generate_sine_wav(normalized_vec_to_orig(orig_contour_downsampled), frame_rate=10, sr=16000)
        soundfile.write(out_path+f"_orig_contour.wav", orig_contour_wav, samplerate=16000)

        #7 save contour plot
        plt.figure()
        plot_contour_with_voice_only(humm_contour_downsampled)
        plot_contour_with_voice_only(orig_contour_downsampled)
        plot_contour_with_voice_only(rec_contour_downsampled)
        plt.legend(["Humm", "Orig", "Rec"])
        plt.savefig(out_path+"_contour_plot.png")
        plt.close()