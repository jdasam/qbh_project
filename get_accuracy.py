import os
import time
import argparse
import _pickle as pickle
import copy

from math import inf as mathinf
from simplejson import dump as json_dump, load as json_load
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
# from adamp import AdamP

from train import prepare_humming_db_loaders,load_hparams
from validation import get_contour_embeddings
from inference import  load_model, load_checkpoint


def validate(model, hparams):
    """Handles all the validation scoring and printing"""
    test_loader, entire_loader = prepare_humming_db_loaders(hparams, return_test=True)
    model.eval()
    num_correct_answer = 0
    with torch.no_grad():
        total_embs, total_song_ids = get_contour_embeddings(model, entire_loader)
        for j, batch in enumerate(test_loader):
            contours, song_ids = batch
            anchor = model(contours.cuda())
            anchor_norm = anchor / anchor.norm(dim=1)[:, None]
            similarity = torch.mm(anchor_norm, total_embs.transpose(0,1))
            recommends = torch.topk(similarity, k=10, dim=-1)[1]
            recommends = total_song_ids[recommends]
            top10_success = [1 for i in range(recommends.shape[0])if int(song_ids[i]) in recommends[i,:].tolist()]
            num_correct_answer += sum(top10_success)
    print(f"N of question: {len(test_loader.dataset)}, N of correct answer:{num_correct_answer}, correct_ratio: {num_correct_answer/len(test_loader.dataset)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        default="/home/svcapp/userdata/flo_model/",
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, 
                    default='/home/svcapp/t2meta/qbh_model/worker_362844_contour_scheduled_hidden256_lr0.0001_210205-133748/checkpoint_best.pt',
                        required=False, help='checkpoint path')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = load_hparams(args.checkpoint_path)
    hparams.humming_path = "/home/svcapp/t2meta/flo_melody/humming_db_contour_pairs.dat"
    hparams.contour_path = "/home/svcapp/t2meta/flo_melody/overlapped.dat"

    model = load_model(hparams)
    model = load_checkpoint(args.checkpoint_path, model)
    model.to('cuda')

    validate(model, hparams)
    