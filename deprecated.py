

### from data_utils.py
class ContourSet:
    def __init__(self, path, song_ids=[], num_aug_samples=4, num_neg_samples=4, quantized=True, pre_load=False, set_type='entire', min_aug=1):
        if not pre_load:
            self.path = Path(path)
            self.melody_txt_list = [song_id_to_pitch_txt_path(self.path, x) for x in song_ids]
            self.melody_loader = MelodyLoader(in_midi_pitch=True, is_quantized=quantized)

            self.contours = self.load_melody()
            # self.pitch_mean, self.pitch_std = get_pitch_stat(self.contours)
            # print(self.pitch_mean, self.pitch_std)
            # for i in tqdm(range(len(self.contours))):
            #     cont = self.contours[i]
            #     norm_cont = normalize_contour(cont['melody'], self.pitch_mean, self.pitch_std)
            #     self.contours[i] = {'melody':norm_cont, 'is_vocal':(np.asarray(cont['melody'])!=0).tolist(),'song_id':cont['song_id'], 'frame_pos':cont['frame_pos']}
            # norm_contours = [normalize_contour(x['melody'], self.pitch_mean, self.pitch_std) for x in contours]
            # norm_contours = [normalize_contour(x[0], self.pitch_mean, self.pitch_std) for x in self.contours]
            # self.contours = [((y, (np.asarray(x['melody'])!=0).tolist(),  x['song_id'], x['frame_pos'] for x,y in zip(self.contours, norm_contours)]
            # generator = ({'melody':y, 'is_vocal':(np.asarray(x['melody'])!=0).tolist(),'song_id':x['song_id'], 'frame_pos':x['frame_pos']} for x,y in zip(contours, norm_contours))
            # self.contours = []
            # for i in generator:
            #     self.contours.append(i)

        else:
            self.contours = path
        self.num_neg_samples = num_neg_samples
        self.num_aug_samples = num_aug_samples
        self.aug_keys = ['tempo', 'key', 'std', 'masking', 'pitch_noise', 'fill', 'drop_out']
        # self.aug_types = ['different_tempo', 'different_key']
        self.down_f = 10
        self.set_type = set_type
        self.min_aug = min_aug

        if set_type =='train':
            self.contours = self.contours[:int(len(self)*0.8)]
        elif set_type =='valid':
            self.contours = self.contours[int(len(self)*0.8):int(len(self)*0.9)]
        elif set_type == 'test':
            self.contours = self.contours[int(len(self)*0.9):]

    def load_melody(self):
        # melody_txt_list = self.path.rglob('*.txt')
        # contours = [self.melody_loader.get_split_contour(txt) for txt in tqdm(self.melody_txt_list)]
        contours = [self.melody_loader.get_overlapped_contours(txt) for txt in tqdm(self.melody_txt_list)]
        contours = [x for x in contours if x is not None]
        contours = [y for x in contours for y in x]
        return contours
    
    def save_melody(self, out_path):
        # contours = self.load_melody()
        # with open(out_path, 'w') as f:
        #     json.dump(self.contours, f)
        with open(out_path, 'wb') as f:
            pickle.dump(self.contours, f)

    def __len__(self):
        return len(self.contours)

    def __getitem__(self, index):
        """
        for training:
        return: (downsampled_melody, [augmented_melodies], [negative_sampled_melodies])
        for validation:
        return: ([augmented_melodies], [selected_song_id])
        """
        selected_melody = self.contours[index]['melody']
        selected_is_vocal = self.contours[index]['is_vocal']
        selected_song_id = self.contours[index]['song_id']
        downsampled_melody = downsample_contour(selected_melody, selected_is_vocal)

        if self.set_type == 'entire':
            return downsampled_melody, selected_song_id

        aug_samples = []
        neg_samples = []
        
        # augmenting melodies
        melody_array = mel_aug.melody_dict_to_array(self.contours[index])
        if self.min_aug < len(self.aug_keys):
            aug_samples = [mel_aug.make_augmented_melody(melody_array, random.sample(self.aug_keys, random.randint(self.min_aug,len(self.aug_keys)))) for i in range(self.num_aug_samples)]
        else:
            aug_samples = [mel_aug.make_augmented_melody(melody_array,self.aug_keys) for i in range(self.num_aug_samples)]
        # if len(self.aug_types) <= self.num_aug_samples:
        #     sampled_aug_types = self.aug_types
        # else:
        #     sampled_aug_types = random.sample(self.aug_types, self.num_aug_samples)
        # for aug_type in sampled_aug_types:
        #     if aug_type == 'different_tempo':
        #         aug_melody = getattr(mel_aug, 'with_different_tempo')(selected_melody, selected_is_vocal)
        #     else:
        #         func = getattr(mel_aug, 'with_'+aug_type)
        #         aug_melody = func(downsampled_melody)
        #     aug_samples.append(aug_melody)
        
        if self.set_type == 'valid':
            return aug_samples, [selected_song_id] * len(aug_samples)
            # return [downsampled_melody] * len(aug_samples), [selected_song_id] * len(aug_samples)

        # sampling negative melodies
        while len(neg_samples) < self.num_neg_samples:
            neg_idx = random.randint(0, len(self)-1)
            if self.contours[neg_idx]['song_id'] != selected_song_id:
                neg_samples.append(downsample_contour(self.contours[neg_idx]['melody'], self.contours[neg_idx]['is_vocal'], self.down_f))
        return downsampled_melody, aug_samples, neg_samples


def normalize_contour(contour, mean, std):
    return [normalize_value_if_not_zero(x, mean, std) for x in contour]

def normalize_value_if_not_zero(value, mean, std):
    if value == 0:
        return value
    else:
        return (value-mean) / std

def get_pitch_stat(contours):
    pitch_values = [y for x in contours for y in x['melody'] if y!=0]
    mean = np.mean(pitch_values)
    std = np.std(pitch_values)
    return mean, std


def quantizing_hz(contour, to_midi=False, quantization=True):
    if quantization is False and to_midi is False:
        return contour
    def quantize_or_return_zero(pitch):
        if pitch > 0:
            if to_midi:
                return hz_to_midi_pitch(pitch, quantization)
            else:
                return 440 * (2 ** ((round(log2(pitch/440) * 12))/12))
        else:
            return 0
    return [quantize_or_return_zero(x) for x in contour]


def hz_to_midi_pitch(hz, quantization=True):
    if hz == 0:
        return 0
    if quantization:
        return round(log2(hz/440) * 12) + 69
    else:
        return log2(hz/440) * 12 + 69