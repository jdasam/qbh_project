from pathlib import Path
import shutil
import os
import tqdm
# pitch_dir = Path('/home/svcapp/userdata/flo_pitch/')
# pitch_list = pitch_dir.rglob("*.txt")
# target_dir = '/home/svcapp/t2meta/flo_new_music/music_100k/'

# for pitch in pitch_list:
#     rel_path = '/'.join(str(pitch.parent).split('/')[-2:]) + '/'
#     if 'qbh' in rel_path:
#         rel_path = 'qbh/'
#     target_path = target_dir + rel_path + pitch.name
#     # print(pitch, target_path)
#     shutil.copyfile(pitch, target_path)


audio_dir = Path('/home/svcapp/t2meta/flo_new_music/music_120k/')
pitch_list = list(audio_dir.rglob("*.m4a")) + list(audio_dir.rglob("*.aac")) + list(audio_dir.rglob("*.txt"))
target_dir = '/home/svcapp/t2meta/flo_new_music/music_100k/'

for pitch in tqdm.tqdm(pitch_list):
    rel_path = '/'.join(str(pitch.parent).split('/')[-2:]) + '/'
    target_path = target_dir + rel_path + pitch.name
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.move(pitch, target_path)