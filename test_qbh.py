
from pathlib import Path
import _pickle as pickle
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import argparse
import copy
import samplerate

from train import load_checkpoint
from model.model import CnnEncoder, CombinedModel
from utils.data_utils import WindowedContourSet, ContourCollate, HummingPairSet, get_song_ids_of_selected_genre, AudioTestSet, AudioCollate, load_audio_sample
from utils.data_path_utils import song_id_to_audio_path
from model.validation import get_contour_embeddings
from utils.melody_utils import MelodyLoader, melody_to_formatted_array
from utils.sampling_utils import downsample_contour_array
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import utils.humming_data_utils as humm_utils
import utils.monitoring as monitoring
from tqdm.auto import tqdm
from model import hparams

import sys
sys.modules['hparams'] = hparams


class QbhSystem:
    def __init__(self, ckpt_dir, emb_dir, device='cuda', audio_dir=None, meta_path='data/flo_metadata_220k.dat', min_vocal_ratio=0.3, make_emb=False, song_ids=[]):
        '''
        ckpt_dir: (str) directory path of checkpoint
        emb_dir: (str) directory path to (save or load) melody embeddings
        device: (str) cpu or cuda
        audio_dir: (str) directory path of audio files (optional)
        meta_path: (str) path to meta dat

        '''

        self.model, self.hparams = load_model(ckpt_dir, device)
        self.model.eval()
        if hasattr(self.hparams, 'end_to_end') and self.hparams.end_to_end:
            self.end_to_end = True
            self.sample_rate = 8000
        else:
            self.end_to_end = False
        self.device = device
        self.load_meta(meta_path)
        self.audio_dir = audio_dir

        if audio_dir and not self.end_to_end:
            self.melody_loader = MelodyLoader(audio_dir, min_ratio=min_vocal_ratio)

        if make_emb:
            self.get_and_save_embedding(song_ids, emb_dir)
        self.load_embedding(emb_dir)

    def load_meta(self, meta_path):
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        self.db_meta = {x['track_id']: x for x in metadata}

    def load_embedding(self, emb_dir):
        pt_lists = Path(emb_dir).rglob('*.pt')
        embs = [torch.load(path) for path in pt_lists]
        self.embedding = torch.cat([x['embedding'] for x in embs]).to(self.device)
        self.embedding /= self.embedding.norm(dim=1)[:,None]
        self.song_ids = torch.LongTensor([x['song_id'] for x in embs for _ in range(x['embedding'].shape[0])])
        self.slice_pos = torch.cat([x['frame_pos'] for x in embs])
        self.unique_ids, self.index_by_id = get_index_by_id(self.song_ids)
        self.slice_by_song = self.slice_pos[self.index_by_id]

    def get_and_save_embedding(self, db_song_ids, save_dir):
        save_dir = Path(save_dir)
        with torch.no_grad():
            for song_id in tqdm(db_song_ids):
                if self.end_to_end:
                    raise NotImplementedError
                    audio_path = song_id_to_audio_path(self.audio_dir, song_id)
                    audio_samples = load_audio_sample(audio_path, self.sample_rate)
                    audio_samples[frame_pos[0]//100*self.sample_rate:frame_pos[1]//100*self.sample_rate]
                else:
                    melodies = self.melody_loader(song_id)
                    if len(melodies) == 0:
                        continue
                    batch = torch.Tensor([downsample_contour_array(x['contour']) for x in melodies]).to(self.device)
                emb = self.model(batch)
                emb /= emb.norm(dim=1)[:,None]
                str_id = str(song_id)
                parent_dir = save_dir/str_id[:3]/str_id[3:6]
                parent_dir.mkdir(parents=True, exist_ok=True)
                output = {'embedding': emb.cpu(), 'song_id': song_id, 'frame_pos':torch.LongTensor([x['frame_pos'] for x in melodies])}
                torch.save(output, save_dir/str_id[:3]/str_id[3:6]/(str_id+'.pt'))

    def get_rec_by_embedding(self, embedding, k=5):
        similarity = cal_similarity(self.embedding, embedding)
        recommends, selected_max_slice_pos, top_k_similarity, _ = get_most_similar_result(similarity, self.unique_ids, self.index_by_id, self.slice_by_song, k=k)
        rec_result = [{'artist': self.db_meta[x]['artist_name_basket'], 'title': self.db_meta[x]['track_name'],'song_id': x, 'slice_pos':y.tolist(), 'similarity':z} 
                            for x,y,z in zip(recommends.squeeze().tolist(), selected_max_slice_pos, top_k_similarity[:k].squeeze().tolist())]
        return rec_result

    def get_rec_by_melody(self, input_melody, k=5):
        if len(input_melody.shape) == 2:
            input_melody = input_melody.unsqueeze(0)
        anchor = self.model(input_melody.to(self.device))
        return self.get_rec_by_embedding(anchor, k=k)

    def get_rec_by_audio(self, audio_sample, sample_rate=44100, k=5):
        if self.end_to_end:
            raise NotImplementedError
        else:
            if sample_rate != 8000:
                audio_sample = samplerate.resample(audio_sample, 8000 / sample_rate, 'sinc_best')
            melody = self.melody_loader.melody_extractor.get_melody_from_audio(audio_sample)
            melody_array = melody_to_formatted_array(melody)
            melody_array = downsample_contour_array(melody_array)
            melody_tensor = torch.Tensor(melody_array).unsqueeze(0)
            anchor = self.model(melody_tensor.to(self.device))
        return self.get_rec_by_embedding(anchor, k=k)

class WrappedModel(torch.nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module 


def cal_similarity(total, emb):
    norm = emb / emb.norm(dim=1)[:, None]
    similarity = torch.mm(norm, total.transpose(0,1))
    return similarity

def get_most_similar_result(similarity, unique_ids, index_by_id, slice_pos_info, k=30):
    # get most similar song id
    similarity_by_ids = similarity[:, index_by_id]
    max_similarity_by_song, max_ids = torch.max(similarity_by_ids, dim=-1)
    top_k_similarity, rec_ids_position = torch.topk(max_similarity_by_song, k=k, dim=-1)
    recommends = unique_ids[rec_ids_position]

    # get slice position of recommendation
    rec_total_slice = slice_pos_info[rec_ids_position.cpu().numpy().squeeze()]
    rec_max_ids = max_ids[0, rec_ids_position].cpu().numpy().squeeze()
    selected_max_slice_pos = [rec_total_slice[i, idx] for i, idx in enumerate(rec_max_ids) ]
    return recommends, selected_max_slice_pos, top_k_similarity, max_similarity_by_song


def cal_rec_rank_of_true_id(similarity, max_similarity_by_song, total_song_ids, song_id):
    corresp_melody_ids = torch.where(total_song_ids==song_id)[0]
    if len(corresp_melody_ids) ==0:
        max_similarity = -1
    else:
        max_similarity = torch.max(similarity[:, corresp_melody_ids])
    max_rank = torch.sum(max_similarity_by_song > max_similarity)
    return max_rank, max_similarity

def load_model(ckpt_dir, device='cuda'):
    model_path = Path(ckpt_dir)
    # hparams = load_hparams(model_path / 'hparams.dat')
    with open(model_path / 'hparams.dat', 'rb') as f:
        hparams = pickle.load(f)
    if hasattr(hparams, 'end_to_end') and hparams.end_to_end:
        hparams_b = copy.copy(hparams)
        hparams_b.input_size = 512
        model = CombinedModel(hparams, hparams_b)
    else:
        model = CnnEncoder(hparams)
    try:
        model, _, _, _ = load_checkpoint(model_path/'checkpoint_best.pt', model, None, train_on_humming=True)
    except:
        model = WrappedModel(model)
        model, _, _, _ = load_checkpoint(model_path/'checkpoint_best.pt', model, None, train_on_humming=True)
        model = model.module
    model = model.to(device)
    return model, hparams


def prepare_humming_testset(humm_contour_pairs_dat_path='/home/svcapp/userdata/flo_melody/humming_db_contour_pairs.dat', num_workers=4):
    with open(humm_contour_pairs_dat_path, 'rb') as f:
        contour_pairs = pickle.load(f)

    humm_test_set = HummingPairSet(contour_pairs, [], "test",[], num_aug_samples=0, num_neg_samples=0)
    humm_test_loader = DataLoader(humm_test_set, 1, shuffle=False,num_workers=num_workers,
        collate_fn=ContourCollate(0, 0, for_cnn=True), pin_memory=True, drop_last=False)

    return humm_test_loader

def prepare_dataset_for_test(data_dir='/home/svcapp/userdata/flo_data_backup/', selected_genres=[4, 12, 13, 17, 10, 7,15, 11, 9], dataset='data/flo_metadata.dat', num_workers=4, min_vocal_ratio=0.5, use_audio=False, sample_rate=8000):

    with open(dataset, 'rb') as f:
        metadata = pickle.load(f)
    with open('data/humm_db_ids.dat', 'rb') as f:
        humm_ids = pickle.load(f)

    song_ids = get_song_ids_of_selected_genre(metadata, selected_genre=selected_genres)
    song_ids += humm_ids
    # song_ids = humm_ids
    # song_ids = [427396913, 5466183, 30894451, 421311716, 420497440]
    if use_audio:
        entireset = AudioTestSet(data_dir, song_ids=song_ids, sample_rate=sample_rate)
        entire_loader = DataLoader(entireset, 4, shuffle=False,num_workers=num_workers,
            collate_fn=AudioCollate(), pin_memory=False, drop_last=False)
    else:
        entireset = WindowedContourSet(data_dir, aug_weights=[], song_ids=song_ids, set_type='entire', num_aug_samples=0, num_neg_samples=0, min_vocal_ratio=min_vocal_ratio)
        entire_loader = DataLoader(entireset, 512, shuffle=False, num_workers=num_workers,
            collate_fn=ContourCollate(0, 0, for_cnn=True), pin_memory=False, drop_last=False)

    humm_test_loader = prepare_humming_testset('/home/svcapp/userdata/flo_melody/humming_db_contour_pairs.dat', num_workers)
    selected_100, selected_900 = humm_utils.load_meta_from_excel("/home/svcapp/userdata/humming_db/Spec.xlsx")

    meta_in_song_key = {x['track_id']: x for x in metadata}
    for song in selected_100.to_dict('records'):
        meta_in_song_key[song['track_id']] = song
    for song in selected_900.to_dict('records'):
        meta_in_song_key[song['track_id']] = song
    return entire_loader, humm_test_loader, meta_in_song_key

def evaluate_single_sample(model, contour, song_id, total_embs, total_song_ids, unique_ids, index_by_id, total_slice_pos_by_song, top_k=10):
    # This function is work in progress
    if len(contour.shape) == 2:
        contour = contour.unsqueeze(0)
    anchor = model(contour.cuda())
    anchor_norm = anchor / anchor.norm(dim=1)[:, None]
    similarity = torch.mm(anchor_norm, total_embs.transpose(0,1))
    similarity_by_ids = similarity[:, index_by_id]
    max_similarity_by_song, max_ids = torch.max(similarity_by_ids, dim=-1)
    corresp_melody_ids = torch.where(total_song_ids==song_id)[0]
    if len(corresp_melody_ids) ==0:
        max_similarity = -1
    else:
        max_similarity = torch.max(similarity[:, corresp_melody_ids])
    max_rank = torch.sum(max_similarity_by_song > max_similarity)
    top10_success = max_rank < 10
    rec_ids_position = torch.topk(max_similarity_by_song, k=30, dim=-1)[1]
    recommends = unique_ids[rec_ids_position]

    rec_total_slice = total_slice_pos_by_song[rec_ids_position.cpu().numpy().squeeze()]
    rec_max_ids = max_ids[0, rec_ids_position].cpu().numpy().squeeze()
    selected_max_slice_pos = [rec_total_slice[i, idx] for i, idx in enumerate(rec_max_ids) ]

    return max_rank, recommends, selected_max_slice_pos

def evaluate(model, humm_test_loader, total_embs, total_song_ids, unique_ids, index_by_id, total_slice_pos, top_k=10):
    model.eval()
    num_correct_answer = 0
    total_recommends = []
    total_test_ids = []
    total_rank = []
    total_rec_slices = []
    total_slice_pos_by_song = total_slice_pos[index_by_id]
    total_corresp_similarity = []
    with torch.no_grad():
    #     total_embs, total_song_ids = get_contour_embeddings(model, entire_loader)
        for j, batch in enumerate(humm_test_loader):
            contours, song_ids = batch
            anchor = model(contours.cuda())

            similarity = cal_similarity(total_embs, anchor)
            recommends, selected_max_slice_pos, _, max_similarity_by_song = get_most_similar_result(similarity, unique_ids, index_by_id, total_slice_pos_by_song, k=30)
            max_rank, max_similarity = cal_rec_rank_of_true_id(similarity, max_similarity_by_song, total_song_ids, song_ids)
            # corresp_melody_ids = torch.where(total_song_ids==song_ids)[0]
            # if len(corresp_melody_ids) ==0:
            #     max_similarity = -1
            # else:
            #     max_similarity = torch.max(similarity[:, corresp_melody_ids])
            # max_rank = torch.sum(max_similarity_by_song > max_similarity)

            top10_success =torch.sum(max_rank<top_k).item()
            num_correct_answer += top10_success
            # top10_success = [ int(int(song_ids[i]) in recommends[i,:top_k].tolist()) for i in range(recommends.shape[0])]
            total_recommends.append(recommends)
            total_test_ids.append(song_ids)
            total_rank.append(max_rank.item())
            total_corresp_similarity.append(max_similarity)
            total_rec_slices.append(selected_max_slice_pos)

    score = num_correct_answer / len(humm_test_loader.dataset)
    print(f'Top {top_k} accuracy: {score}')
    total_recommends = torch.cat(total_recommends, dim=0).cpu().numpy()
    total_test_ids = torch.cat(total_test_ids, dim=0).cpu().numpy()
    total_rec_slices = np.asarray(total_rec_slices)
    total_corresp_similarity = np.asarray(total_corresp_similarity)
    mrr_score = np.mean(1 / (np.asarray(total_rank)+1))
    print('MRR score: ', mrr_score)
    return score, mrr_score, total_recommends, total_test_ids, total_rank, total_rec_slices, total_corresp_similarity

def get_index_by_id(total_song_ids):
    out = []
    unique_ids = list(set(total_song_ids.tolist()))
    for id in unique_ids:
        out.append(torch.where(total_song_ids==id)[0])
    max_len = max([len(x) for x in out])
    dummy = torch.zeros((len(unique_ids), max_len), dtype=torch.long)
    for i, ids in enumerate(out):
        dummy[i,:len(ids)] = ids
        dummy[i, len(ids):] = ids[-1]
    return torch.LongTensor(unique_ids), dummy

def convert_result_to_dict(ids, ranks, meta):
    out = defaultdict(list)
    for id, r in zip(ids, ranks):
        out[meta[id]['artist_name'] + ' - ' + meta[id]['track_name']].append(r)
    return dict(out)


def convert_result_to_rec_title(total_test_ids, total_recommends, total_rank, total_corresp_similarity, meta, humm_meta, k=3):
    out = {}
    for idx in total_test_ids:
        out[meta[idx]['artist_name'] + ' - ' + meta[idx]['track_name']] = [idx] + [ [] for i in range(5)]
    
    for idx, rec, r, sim, humm in zip(total_test_ids, total_recommends, total_rank, total_corresp_similarity, humm_meta):
        target = out[meta[idx]['artist_name'] + ' - ' + meta[idx]['track_name']]
        string =  "\n".join([f'Rec rank: {r+1}'] + ["Similarity with Orig: {:.4f}".format(sim)] + [monitoring.id_to_name(idx, meta) for idx in rec[:k]]
                            + [f'Group: {humm["singer_group"]}', f'Singer ID: {humm["singer_id"]}', f'Gender: {humm["singer_gender"]}', f'Humm type: {humm["humming_type"]}'])
        if humm['singer_group'] == 'P':
            if target[1] == []:
                target[1] = string
            else:
                target[2] =  string
        else:
            if target[3] ==[]:
                target[3] =  string
            elif target[4] ==[]:
                target[4] =  string
            else:
                target[5] =  string
    return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-wav', '--save_wav', action='store_true', help="Option for save wav for each evaluation case")
    parser.add_argument('--min_vocal_ratio', type=float, default=0.3)
    parser.add_argument('-data', '--dataset_meta', type=str, default='data/flo_metadata_220k.dat')
    parser.add_argument('--save_dir', type=Path, default='eval/')
    parser.add_argument('--model_dir', type=str, default='/home/svcapp/t2meta/qbh_model')
    args = parser.parse_args()
    if not args.save_dir.exists():
        args.save_dir.mkdir()
    selected_genre = [4]
    entire_loader, humm_test_loader, meta = prepare_dataset_for_test(data_dir='/home/svcapp/t2meta/flo_new_music/music_100k/',  dataset=args.dataset_meta, min_vocal_ratio=args.min_vocal_ratio, selected_genres=selected_genre)
    total_slice_pos = np.asarray([x['frame_pos'] for x in entire_loader.dataset.contours])
    font_path = 'malgun.ttf'
    font_prop = fm.FontProperties(fname=font_path, size=20)

    flo_test_list = pd.read_csv('data/flo_test_list.csv')
    flo_test_meta = {x['track id']: x for x in flo_test_list.to_dict('records')}
    humm_meta = [x['meta'] for x in humm_test_loader.dataset.contours]


    # worker_ids = [401032, 480785, 482492, 482457, 483559, 483461]
    # worker_ids = [483559, 483461]
    worker_ids = [485391] #, 485399]
    # worker_ids = [482492]
    model_dir = Path(args.model_dir)
    for id in worker_ids:
        worker_save_dir = args.save_dir / str(id)
        if not worker_save_dir.exists():
            worker_save_dir.mkdir()
        ckpt_dir = next(model_dir.glob(f"worker_{id}*"))
        model, _ = load_model(ckpt_dir)
        total_embs, total_song_ids = get_contour_embeddings(model, entire_loader)
        unique_ids, index_by_id = get_index_by_id(total_song_ids)
        score, mrr_score, total_recommends, total_test_ids, total_rank, total_rec_slices, total_corresp_similarity = evaluate(model, humm_test_loader, total_embs, total_song_ids, unique_ids, index_by_id, total_slice_pos)
        
        out = convert_result_to_dict(total_test_ids, total_rank, meta)
        detailed_out = convert_result_to_rec_title(total_test_ids, total_recommends, total_rank, total_corresp_similarity, meta, humm_meta)
            
        dataframe = pd.DataFrame(detailed_out).transpose()
        dataframe.insert(1, 'Class', [flo_test_meta[x]['해당 요건'] for x in dataframe[0].values])
        dataframe = dataframe.sort_values('Class')
        sorted_keys = dataframe.to_dict()[0].keys()
        dataframe = dataframe.drop(columns=[0])
        dataframe.to_csv(str(worker_save_dir / f"worker_{id}_87k_eval_table_top10{score}_mrr{mrr_score}.csv"))
        rank_array = np.asarray([out[x] for x in sorted_keys])
        fig = plt.figure(figsize=(20,20))
        ax = plt.gca()
        plt.imshow(1/(rank_array+1))
        plt.colorbar()
        ax.set_yticks(list(range(len(rank_array))))
        ax.set_yticklabels(sorted_keys)
        for label in ax.get_yticklabels() :
            label.set_fontproperties(font_prop)    
        plt.savefig(str(worker_save_dir / f'worker_{id}_87k_eval_matrix.png'))

        if args.save_wav:
            monitoring.save_test_result_in_wav(total_recommends, total_test_ids, total_rank, total_rec_slices, meta, humm_meta, out_dir=worker_save_dir)

# 결과 표에 곡명, 장르별로 정렬, Prof/Non-prof 구별 
