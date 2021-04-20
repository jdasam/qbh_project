
from pathlib import Path
import _pickle as pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict
import numpy as np
import argparse

from train import load_hparams, load_model, load_checkpoint, make_aug_param_dictionary
from model import CnnEncoder, CombinedModel
from data_utils import WindowedContourSet, ContourCollate, HummingPairSet, get_song_ids_of_selected_genre, AudioTestSet, AudioCollate
from validation import get_contour_embeddings, cal_ndcg_single
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random
import pandas as pd
import humming_data_utils as utils
import copy
import monitoring

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module 

def load_model(ckpt_dir, data_parallel=False):
    model_path = Path(ckpt_dir)
    # hparams = load_hparams(model_path / 'hparams.dat')
    with open(model_path / 'hparams.dat', 'rb') as f:
        hparams = pickle.load(f)
    hparams_b = copy.copy(hparams)
    hparams_b.input_size = 512

    if data_parallel:
        model = CombinedModel(hparams, hparams_b)
        model = torch.nn.DataParallel(model)
        model, _, _, _ = load_checkpoint(model_path/'checkpoint_best.pt', model, None, train_on_humming=True)
        model = model.cuda()
    else:
        model = CombinedModel(hparams, hparams_b).cuda()
        model = WrappedModel(model)
        model, _, _, _ = load_checkpoint(model_path/'checkpoint_best.pt', model, None, train_on_humming=True)
        model = model.module
    return model


def prepare_dataset(data_dir='/home/svcapp/userdata/flo_data_backup/', selected_genres=[4, 12, 13, 17, 10, 7,15, 11, 9], dataset='flo_metadata_220k.dat', num_workers=4, batch_size=32):
      
    with open(dataset, 'rb') as f:
        metadata = pickle.load(f)
    with open('humm_db_ids.dat', 'rb') as f:
        humm_ids = pickle.load(f)

    song_ids = get_song_ids_of_selected_genre(metadata, selected_genre=selected_genres)
    song_ids += humm_ids
    song_ids = humm_ids
    # song_ids = [427396913, 5466183, 30894451, 421311716, 420497440]
    # entireset = WindowedContourSet(data_dir, aug_weights=[], song_ids=song_ids, set_type='entire', pre_load=False, num_aug_samples=0, num_neg_samples=0, min_vocal_ratio=min_vocal_ratio)
    entireset = AudioTestSet(data_dir, song_ids)

    entire_loader = DataLoader(entireset, batch_size, shuffle=False,num_workers=num_workers,
        collate_fn=AudioCollate(), pin_memory=False, drop_last=False)

    # with open(hparams.humming_path, "rb") as f:
    with open('/home/svcapp/userdata/flo_melody/humming_db_contour_pairs.dat', 'rb') as f:
        contour_pairs = pickle.load(f)

    humm_test_set = HummingPairSet(contour_pairs, [], "test",[], num_aug_samples=0, num_neg_samples=0)
    humm_test_loader = DataLoader(humm_test_set, 1, shuffle=False,num_workers=num_workers,
        collate_fn=ContourCollate(0, 0, for_cnn=True), pin_memory=True, drop_last=False)

    selected_100, selected_900 = utils.load_meta_from_excel("/home/svcapp/userdata/humming_db/Spec.xlsx")

    meta_in_song_key = {x['track_id']: x for x in metadata}
    for song in selected_100.to_dict('records'):
        meta_in_song_key[song['track_id']] = song
    for song in selected_900.to_dict('records'):
        meta_in_song_key[song['track_id']] = song
    return entire_loader, humm_test_loader, meta_in_song_key


def evaluate(model, humm_test_loader, total_embs, total_song_ids, unique_ids, index_by_id, total_slice_pos):
    model.eval()
    num_correct_answer = 0
    total_success = []
    total_recommends = []
    total_test_ids = []
    total_rank = []
    total_rec_slices = []
    total_slice_pos_by_song = total_slice_pos[index_by_id]

    with torch.no_grad():
    #     total_embs, total_song_ids = get_contour_embeddings(model, entire_loader)
        for j, batch in enumerate(humm_test_loader):
            contours, song_ids = batch
            anchor = model(contours.cuda(), contour_only=True)
            anchor_norm = anchor / anchor.norm(dim=1)[:, None]
            similarity = torch.mm(anchor_norm, total_embs.transpose(0,1))
            similarity_by_ids = similarity[:, index_by_id]
            max_similarity_by_song, max_ids = torch.max(similarity_by_ids, dim=-1)

            corresp_melody_ids = torch.where(total_song_ids==song_ids)[0]
            if len(corresp_melody_ids) ==0:
                max_similarity = -1
            else:
                max_similarity = torch.max(similarity[:, corresp_melody_ids])
            max_rank = torch.sum(max_similarity_by_song > max_similarity)
            rec_ids_position = torch.topk(max_similarity_by_song, k=30, dim=-1)[1]
            recommends = unique_ids[rec_ids_position]
            top10_success = [ int(int(song_ids[i]) in recommends[i,:10].tolist()) for i in range(recommends.shape[0])]
            total_success += top10_success
            total_recommends.append(recommends)
            total_test_ids.append(song_ids)
            total_rank.append(max_rank.item())

            rec_total_slice = total_slice_pos_by_song[rec_ids_position.cpu().numpy().squeeze()]
            rec_max_ids = max_ids[0, rec_ids_position].cpu().numpy().squeeze()
            selected_max_slice_pos = [rec_total_slice[i, idx] for i, idx in enumerate(rec_max_ids) ]
            total_rec_slices.append(selected_max_slice_pos)

            
            num_correct_answer += sum(top10_success)
    score = num_correct_answer / len(humm_test_loader.dataset)
    print(score)
    mrr_score = np.mean(1 / (np.asarray(total_rank)+1))
    print('mrr: ', mrr_score)
    total_recommends = torch.cat(total_recommends, dim=0).cpu().numpy()
    total_test_ids = torch.cat(total_test_ids, dim=0).cpu().numpy()
    total_rec_slices = np.asarray(total_rec_slices)

    return score, mrr_score, total_recommends, total_test_ids, total_rank, total_rec_slices

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

def get_similarity_by_id(similarity, unique_ids, index_by_ids):
    return

def convert_result_to_dict(ids, ranks, meta):
    out = defaultdict(list)
    for id, r in zip(ids, ranks):
        out[meta[id]['artist_name'] + ' - ' + meta[id]['track_name']].append(r)
    return dict(out)


def convert_result_to_rec_title(total_test_ids, total_recommends, total_rank, meta, humm_meta, k=3):
    out = {}
    for idx in total_test_ids:
        out[meta[idx]['artist_name'] + ' - ' + meta[idx]['track_name']] = [idx] + [ [] for i in range(5)]
    
    for idx, rec, r, humm in zip(total_test_ids, total_recommends, total_rank, humm_meta):
        target = out[meta[idx]['artist_name'] + ' - ' + meta[idx]['track_name']]
        string =  "\n".join([f'Rec rank: {r+1}'] + [monitoring.id_to_name(idx, meta) for idx in rec[:k]]
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


def save_dict_result_to_csv(adict):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-wav', '--save_wav', action='store_true', help="Option for save wav for each evaluation case")
    parser.add_argument('--min_vocal_ratio', type=float, default=0.3)
    parser.add_argument('-data', '--dataset_meta', type=str, default='flo_metadata_220k.dat')
    parser.add_argument('--save_dir', type=Path, default='eval/')
    parser.add_argument('--model_dir', type=str, default='/home/svcapp/t2meta/qbh_model')
    parser.add_argument('--data_dir', type=str, default='/home/svcapp/t2meta/flo_new_music/music_100k/' )
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_parallel', action='store_true')

    args = parser.parse_args()
    if not args.save_dir.exists():
        args.save_dir.mkdir()
    entire_loader, humm_test_loader, meta = prepare_dataset(data_dir=args.data_dir,  dataset=args.dataset_meta, num_workers=args.num_workers)
    total_slice_pos = np.asarray(entire_loader.dataset.slice_infos)

    # entire_loader, humm_test_loader, meta = prepare_dataset(data_dir='/home/svcapp/t2meta/flo_new_music/music_100k/', batch_size=64, num_workers=4)

    font_path = 'malgun.ttf'
    font_prop = fm.FontProperties(fname=font_path, size=20)

    flo_test_list = pd.read_csv('flo_test_list.csv')
    flo_test_meta = {x['track id ']: x for x in flo_test_list.to_dict('records')}
    humm_meta = [x['meta'] for x in humm_test_loader.dataset.contours]


    worker_ids = [484078, 484075]
    model_dir = Path('/home/svcapp/t2meta/end-to-end-qbh')
    for id in worker_ids:
        worker_save_dir = args.save_dir / str(id)
        if not worker_save_dir.exists():
            worker_save_dir.mkdir()

        ckpt_dir = next(model_dir.glob(f"worker_{id}*"))
        model = load_model(ckpt_dir, args.data_parallel)
        total_embs, total_song_ids = get_contour_embeddings(model, entire_loader)
        unique_ids, index_by_id = get_index_by_id(total_song_ids)
        score, mrr_score, total_recommends, total_test_ids, total_rank, total_rec_slices = evaluate(model, humm_test_loader, total_embs, total_song_ids, unique_ids, index_by_id, total_slice_pos)
        
        out = convert_result_to_dict(total_test_ids, total_rank, meta)
        detailed_out = convert_result_to_rec_title(total_test_ids, total_recommends, total_rank, meta, humm_meta)

            
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
