from sampling_utils import downsample_contour_array
import torch
from torch.utils.data import DataLoader
from math import log
from sampling_utils import downsample_contour_array

def get_contour_embs_from_overlapped_contours(model, dataset, batch_size=128):
    total_embs = torch.zeros([len(dataset), model.embed_size]).to('cuda')
    # total_song_ids = torch.zeros(len(dataset),dtype=torch.long)
    total_song_ids = []
    current_idx = 0
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = torch.Tensor([downsample_contour_array(x['contour']) for x in dataset[i:i+batch_size]]).cuda()
            song_ids = [x['song_id'] for x in dataset[i:i+batch_size]]
            # song_ids = torch.Tensor([x['song_id'] for x in dataset[i:i+batch_size]])
            embeddings = model(batch)
            num_samples = len(song_ids)
            total_embs[i:i+num_samples,:] = embeddings / embeddings.norm(dim=1)[:,None]
            total_song_ids += song_ids

    return total_embs, total_song_ids


def get_contour_embeddings(model, cmp_loader):
    # cmp_loader: loading entire trainset to calculate embedding of each piece
    if hasattr(model, 'embed_size'):
        embed_size = model.embed_size
    else:
        embed_size = model.module.embed_size
    total_embs = torch.zeros([len(cmp_loader.dataset), embed_size]).to('cuda')
    total_song_ids = torch.zeros(len(cmp_loader.dataset),dtype=torch.long)
    current_idx = 0
    model.eval()
    with torch.no_grad():
        for batch in cmp_loader:
            audio, song_ids = batch
            # embeddings = model(contour.cuda())
            # num_batch = audio.shape[0]
            embeddings = model(audio.cuda())
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
    if rel_recs == []:
        return 0
    else:
        return rel_recs[0]
    # dcg = sum(rel_recs)
    # return dcg