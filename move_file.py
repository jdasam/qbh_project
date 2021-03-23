from pathlib import Path
import shutil

pitch_dir = Path('/home/svcapp/userdata/flo_pitch/')
pitch_list = pitch_dir.rglob("*.txt")
target_dir = '/home/svcapp/t2meta/flo_new_music/music_100k/'

for pitch in pitch_list:
    rel_path = '/'.join(str(pitch.parent).split('/')[-2:]) + '/'
    if 'qbh' in rel_path:
        rel_path = 'qbh/'
    target_path = target_dir + rel_path + pitch.name
    # print(pitch, target_path)
    shutil.copyfile(pitch, target_path)