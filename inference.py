import os
import time
import argparse
import pickle

from math import inf as mathinf
from simplejson import dump as json_dump, load as json_load
import numpy as np
from tqdm import tqdm
from numpy import finfo
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model import ContourEncoder, CnnEncoder
from data_utils import ContourSet, ContourCollate, pad_collate
from torch.optim.lr_scheduler import StepLR
from logger import Logger
from hparams import HParams
from loss_function import SiameseLoss
from validation import get_contour_embeddings, cal_ndcg, cal_ndcg_single, get_contour_embs_from_overlapped_contours


def prepare_dataloaders(hparams, contour_path):
    # Get data, data loaders and collate function ready
    with open(contour_path, 'rb') as f:
        # pre_loaded_data = json_load(f)
        pre_loaded_data = pickle.load(f)
    entireset = ContourSet(pre_loaded_data, set_type='entire', pre_load=True, num_aug_samples=0, num_neg_samples=0)
    entire_loader = DataLoader(entireset, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
        collate_fn=ContourCollate(0, 0, for_cnn=True), pin_memory=True, drop_last=False)

    return entire_loader

def prepare_directories_and_logger(output_directory, log_directory,):
    print(output_directory, log_directory)
    logger = Logger(output_directory / log_directory)
    return logger

def load_model(hparams):
    # model = ContourEncoder(hparams).cuda()
    model = CnnEncoder(hparams).cuda()
    if hparams.data_parallel:
        model = torch.nn.DataParallel(model)
    return model

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']
    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model


def load_hparams(checkpoint_path):
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    # hparams_path = checkpoint_path.parent / 'hparams.json'
    # with open(hparams_path) as json_file:
    #     return AttributeDict(json_load(json_file))
    hparams_path = checkpoint_path.parent / 'hparams.dat'
    with open(hparams_path, 'rb') as f:
        return pickle.load(f)

def convert_hparams_to_string(hparams):
    return '{}_hidden{}_lr{}_{}/'.format(
        hparams.model_code, 
        hparams.hidden_size, 
        hparams.learning_rate, 
        datetime.now().strftime('%y%m%d-%H%M%S')
        )

def validate_classification_error(predicted, answer, threshold=0.6):
    predicted = predicted > threshold
    num_prediction = torch.sum(predicted)
    num_true = torch.sum(answer)
    num_correct = torch.sum(predicted * answer)

    if num_correct == 0:
        return torch.zeros(1)

    precision = num_correct / num_prediction
    recall = num_correct / num_true
    
    return 2* precision * recall / (precision + recall) 



def inference(contour_path, output_directory, checkpoint_path, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """

    model = load_model(hparams)
    # entire_loader = prepare_dataloaders(hparams, contour_path)
    # train_loader,val_loader, cmp_loader, _ = prepare_dataloaders(hparams)

    model = load_checkpoint(checkpoint_path, model)
    with open(contour_path, 'rb') as f:
        dataset = pickle.load(f)
    model.eval()
    # total_embs, total_song_ids = get_contour_embeddings(model, entire_loader)
    total_embs, total_song_ids = get_contour_embs_from_overlapped_contours(model, dataset)
    torch.save({'embs':total_embs.cpu(), 'ids':total_song_ids, 'pos':[x['frame_pos'] for x in dataset]},
               output_directory/"qbh_embedding.pt", )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        default="/home/svcapp/userdata/flo_model/",
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, 
                    default='/home/svcapp/userdata/flo_model/worker_268365_contour_scheduled_hidden256_lr0.0001_201208-100526/checkpoint_last.pt',
                        required=False, help='checkpoint path')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--device', type=int, default=0,
                    required=False, help='gpu device index')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--conv_size', type=int, default=128)        
    parser.add_argument('--out_size', type=int, default=256)        
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)        

    parser.add_argument('--model_code', type=str, default="contour_ae")

    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--encoder_size', type=int, default=32)        
    parser.add_argument('--middle_size', type=int, default=16)        
    parser.add_argument('--latent_vec_size', type=int, default=10)

    args = parser.parse_args()
    if args.checkpoint_path:
        hparams = load_hparams(args.checkpoint_path)
    else:
        # hparams = create_hparams(args.hparams)
        hparams = HParams()
    
    
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    if not hparams.data_parallel and not hparams.in_meta:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    
    output_directory = Path(args.output_directory) 

    inference('/home/svcapp/userdata/flo_melody/overlapped.dat',output_directory, args.checkpoint_path, hparams)
