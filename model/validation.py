import torch
from math import log
from utils.sampling_utils import downsample_contour_array

def get_contour_embs_from_overlapped_contours(model, dataset, batch_size=128):
    try:
        embed_size = model.embed_size
    except:
        embed_size = model.module.embed_size
    total_embs = torch.zeros([len(dataset), embed_size]).to('cuda')
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
    try:
        embed_size = model.embed_size
    except:
        embed_size = model.module.embed_size

    total_embs = torch.zeros([len(cmp_loader.dataset), embed_size]).to('cuda')
    total_song_ids = torch.zeros(len(cmp_loader.dataset),dtype=torch.long)
    current_idx = 0
    model.eval()
    with torch.no_grad():
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
    if rel_recs == []:
        return 0
    else:
        return rel_recs[0]
    # dcg = sum(rel_recs)
    # return dcg


def cal_ndcg_of_loader(model, val_loader, total_embs, total_song_ids, num_recom=50):
    valid_score = 0
    for j, batch in enumerate(val_loader):
        contours, song_ids = batch
        anchor = model(contours.cuda())
        anchor_norm = anchor / anchor.norm(dim=1)[:, None]
        similarity = torch.mm(anchor_norm, total_embs.transpose(0,1))
        recommends = torch.topk(similarity, k=num_recom, dim=-1)[1]
        recommends = total_song_ids[recommends]
        ndcg = [cal_ndcg_single(recommends[i,:], song_ids[i]) for i in range(recommends.shape[0])]
        ndcg = sum(ndcg) / len(ndcg)
        valid_score += ndcg
    valid_score = valid_score/(j+1)
    return valid_score


def cal_mrr_of_loader(model, val_loader, total_embs, total_song_ids):
    valid_score = 0
    for j, batch in enumerate(val_loader):
        contours, song_ids = batch
        anchor = model(contours.cuda())
        anchor_norm = anchor / anchor.norm(dim=1)[:, None]
        similarity = torch.mm(anchor_norm, total_embs.transpose(0,1))
        corresp_melody_ids = [torch.where(total_song_ids==x)[0] for x in song_ids]
        max_corresp_similarities = torch.Tensor([torch.max(similarity[:,x]) for x in corresp_melody_ids])
        search_rank = torch.sum(similarity.cpu() - max_corresp_similarities.unsqueeze(1) > 0, dim=1, dtype=torch.float32)
        mrr_score = 1/(search_rank+1)
        valid_score += torch.mean(mrr_score).item()
    valid_score /= j+1
    return valid_score

