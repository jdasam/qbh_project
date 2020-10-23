import torch
from torch.utils.data import DataLoader
from math import log


def get_contour_embeddings(model, cmp_loader):
    # cmp_loader: loading entire trainset to calculate embedding of each piece
    total_embs = torch.zeros([len(cmp_loader.dataset), model.embed_size]).to('cuda')
    total_song_ids = torch.zeros(len(cmp_loader.dataset),dtype=torch.long)
    current_idx = 0

    for batch in cmp_loader:
        contour, song_ids = batch
        embeddings = model(contour.cuda())
        num_samples = song_ids.shape[0]
        total_embs[current_idx:current_idx+num_samples,:] = embeddings / embeddings.norm(dim=1)[:,None]
        total_song_ids[current_idx:current_idx+num_samples] = song_ids
        current_idx += num_samples

    return total_embs, total_song_ids

def cal_ndcg(rec, answer):
    rel_recs = [ 1/log(i+2,2) for i, value in enumerate(rec) if value in answer]
    dcg = sum(rel_recs)
    # idcg = sum([ 1/log(i+2,2) for i, value in enumerate(answer) if value in answer])
    idcg = sum([ 1/log(i+2,2) for i in range(len(answer))])
    return dcg / idcg

def cal_ndcg_single(rec, answer):
    rel_recs = [ 1/log(i+2,2) for i, value in enumerate(rec) if value == answer]
    dcg = sum(rel_recs)
    return dcg