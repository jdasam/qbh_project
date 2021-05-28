def song_id_to_pitch_txt_path(path, song_id):
        '''
        path: pathlib.Path()
        '''
        txt_path = path / str(song_id)[:3] / str(song_id)[3:6] / '{}_pitch.txt'.format(song_id)
        # txt_path = path / str(song_id)[:3] / str(song_id)[3:6] / 'pitch_{}.txt'.format(song_id)
        if not txt_path.exists():
            txt_path = path / 'qbh' / f'{song_id}_pitch.txt'
        return txt_path
        # return path  / f'pitch_{song_id}.txt'

def song_id_to_audio_path(path, song_id):
    # path: pathlib.Path()
    audio_path = path / str(song_id)[:3] / str(song_id)[3:6] / '{}.aac'.format(song_id)
    if not audio_path.exists():
        audio_path = audio_path.with_suffix('.m4a')
    if not audio_path.exists():
        audio_path = path / 'qbh' / f'{song_id}.aac'
    if not audio_path.exists():
        return None
    else:
        return audio_path
    # return path  / f'pitch_{song_id}.txt'
