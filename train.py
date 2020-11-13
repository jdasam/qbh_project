import os
import time
import argparse
import pickle

from math import inf as mathinf
from simplejson import dump as json_dump, load as json_load
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
# from adamp import AdamP

from model import ContourEncoder, CnnEncoder
from data_utils import ContourSet, ContourCollate, pad_collate
from torch.optim.lr_scheduler import StepLR
from logger import Logger
from hparams import HParams
from loss_function import SiameseLoss
from validation import get_contour_embeddings, cal_ndcg, cal_ndcg_single

from metalearner.common.config import experiment, worker
from metalearner.api import scalars


def prepare_dataloaders(hparams, valid_only=False):
    # Get data, data loaders and collate function ready
    with open(hparams.contour_path, 'rb') as f:
        pre_loaded_data = json_load(f)
    if hparams.is_scheduled:
        min_aug=1
    else:
        min_aug=10
    trainset = ContourSet(pre_loaded_data, set_type='train', pre_load=True, num_aug_samples=hparams.num_pos_samples, num_neg_samples=hparams.num_neg_samples, min_aug=min_aug)
    entireset = ContourSet(pre_loaded_data, set_type='entire', pre_load=True, num_aug_samples=0, num_neg_samples=0)
    validset =  ContourSet(pre_loaded_data, set_type='valid', pre_load=True, num_aug_samples=4, num_neg_samples=0, min_aug=10)

    train_loader = DataLoader(trainset, hparams.batch_size, shuffle=True,num_workers=hparams.num_workers,
        collate_fn=ContourCollate(hparams.num_pos_samples, hparams.num_neg_samples, for_cnn=True), pin_memory=True)
    entire_loader = DataLoader(entireset, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
        collate_fn=ContourCollate(0, 0, for_cnn=True), pin_memory=True, drop_last=False)
    valid_loader = DataLoader(validset, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
        collate_fn=ContourCollate(hparams.num_pos_samples, 0, for_cnn=True), pin_memory=True, drop_last=False)

    return train_loader, valid_loader, entire_loader #, comparison_loader, collate_fn
    # return train_loader, #valid_loader, list_collate_fn

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

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def save_hparams(hparams, output_dir):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_name = output_dir / 'hparams.dat'
    # output_name = output_dir / 'hparams.json'
    # with open(output_name, 'w', encoding='utf-8') as f:
    #     json_dump(hparams.to_json(), f, ensure_ascii=False)
    with open(output_name, 'wb') as f:
        pickle.dump(hparams, f)

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

def cal_ndcg_of_loader(model, val_loader, total_embs, total_song_ids):
    valid_score = 0
    for j, batch in enumerate(val_loader):
        contours, song_ids = batch
        anchor = model(contours.cuda())
        anchor_norm = anchor / anchor.norm(dim=1)[:, None]
        similarity = torch.mm(anchor_norm, total_embs.transpose(0,1))
        recommends = torch.topk(similarity, k=hparams.num_recom, dim=-1)[1]
        recommends = total_song_ids[recommends]
        ndcg = [cal_ndcg_single(recommends[i,:], song_ids[i]) for i in range(recommends.shape[0])]
        ndcg = sum(ndcg) / len(ndcg)
        valid_score += ndcg
    valid_score = valid_score/(j+1)
    return valid_score

def validate(model, val_loader, entire_loader, logger, epoch, iteration, criterion, hparams):
    """Handles all the validation scoring and printing"""
    model.eval()
    valid_score = {}
    with torch.no_grad():
        total_embs, total_song_ids = get_contour_embeddings(model, entire_loader)
        valid_score["validation_score"] = cal_ndcg_of_loader(model, val_loader, total_embs, total_song_ids)
        # for j, batch in enumerate(tqdm(val_loader)):
        if hparams.get_valid_by_aug:
            aug_keys = [x for x in val_loader.dataset.aug_keys]
            for key in aug_keys:
                val_loader.dataset.aug_keys = [key]
                valid_score_of_key = cal_ndcg_of_loader(model, val_loader, total_embs, total_song_ids)
                valid_score[key] = valid_score_of_key
            val_loader.dataset.aug_keys = aug_keys
    model.train()
    if len(valid_score) == 1:
        print("Valdiation nDCG {}: {:5f} ".format(iteration, valid_score["validation_score"]))
    else:
        score_string = "Valdiation nDCG {}: {:5f} ".format(iteration, valid_score["validation_score"])
        for key in valid_score.keys():
            score_string += "/ {}: {:5f}".format(key, valid_score[key])
        print(score_string)
    # if 'siamese' in hparams.model_code:
    #     print("Validation loss {}: {:9f}  ".format(iteration, valid_ndcg))
    # else:
    #     print("Valdiation Score {}: {:5f} ".format(iteration, valid_ndcg))
    #     valid_ndcg = -valid_ndcg
    # audio_sample = valset.convert_spec_to_wav(y_pred.squeeze(0).cpu().numpy())
    if hparams.in_meta:
        # results = {'validation_score': valid_score}
        response = scalars.send_valid_result(worker.id, epoch, iteration, valid_score)
    else:
        logger.log_validation(valid_score, model, iteration)

    return valid_score["validation_score"]

def train(output_directory, log_directory, checkpoint_path, warm_start, hparams):
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

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    # if hparams.optimizer_type.lower() == 'adamp':
    #     optimizer = AdamP(model.parameters(), lr=learning_rate,
    #                                 weight_decay=hparams.weight_decay)
    # else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=hparams.weight_decay)

    logger = prepare_directories_and_logger(output_directory, log_directory)
    save_hparams(hparams, output_directory)
    train_loader, val_loader, entire_loader = prepare_dataloaders(hparams)
    # train_loader,val_loader, cmp_loader, _ = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        # if warm_start:
        #     model = warm_start_model(
        #         checkpoint_path, model, hparams.ignore_layers)
        # else:
        model, optimizer, _learning_rate, iteration = load_checkpoint(
            checkpoint_path, model, optimizer)
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))
    else:
        save_hparams(hparams, output_directory)

    scheduler = StepLR(optimizer, step_size=hparams.learning_rate_decay_steps,
                       gamma=hparams.learning_rate_decay_rate)
    model.train()
    criterion = SiameseLoss(margin=hparams.loss_margin)
    best_valid_score = 0
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        # print("Epoch: {}".format(epoch))
        for _, batch in enumerate(train_loader):
            start = time.perf_counter()
            model.zero_grad()
            batch = batch.cuda()
            anchor, pos, neg = model.siamese(batch)
            loss = criterion(anchor, pos, neg)
            reduced_loss = loss.item()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()
            duration = time.perf_counter() - start
            if hparams.in_meta:
                results = {'training loss': -reduced_loss}
                # response = scalars.send_train_result(worker.id, epoch, iteration, results)
            else:
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)
            if iteration % hparams.iters_per_checkpoint == 0: # and not iteration==0:
                # del loss, batch
                # torch.cuda.empty_cache()
                valid_score = validate(model, val_loader, entire_loader, logger, epoch, iteration, criterion, hparams)
                is_best = valid_score > best_valid_score
                best_valid_score = max(valid_score, best_valid_score)
                if is_best:
                    checkpoint_path = output_directory / 'checkpoint_best'
                    # checkpoint_path = os.path.join(output_directory, "checkpoint_best")
                    # checkpoint_path = os.path.join(
                    #     output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
                else:
                    checkpoint_path = output_directory / 'checkpoint_last'
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
                # torch.cuda.empty_cache()


            iteration += 1
        # train_loader.dataset.min_aug = 1 + iteration // 75000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        default="/home/svcapp/userdata/flo_model/",
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        default = "logdir/",
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--device', type=int, default=0,
                    required=False, help='gpu device index')
    parser.add_argument('--contour_path', type=str,
                    help='path to contour.json')
    parser.add_argument('--in_metalearner', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether work in meta learner')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--hidden_size', type=int, required=False)        
    parser.add_argument('--embed_size', type=int, required=False)
    parser.add_argument('--kernel_size', type=int, required=False)
    parser.add_argument('--num_head', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--valid_batch_size', type=int, required=False)

    parser.add_argument('--epochs', type=int, required=False)    
    parser.add_argument('--iters_per_checkpoint', type=int, required=False)        
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--drop_out', type=float)
    parser.add_argument('--num_workers', type=int)        
    parser.add_argument('--model_code', type=str)
    parser.add_argument('--optimizer_type', type=str)
    parser.add_argument('--num_neg_samples', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--loss_margin', type=float)

    parser.add_argument('--use_attention', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--use_pre_encoder', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--use_rnn', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--is_scheduled', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--get_valid_by_aug', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--use_res', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--use_gradual_size', type=lambda x: (str(x).lower() == 'true'))


    args = parser.parse_args()
    if args.checkpoint_path:
        hparams = load_hparams(args.checkpoint_path)
    else:
        # hparams = create_hparams(args.hparams)
        hparams = HParams()
    
    
    if args.in_metalearner:
        hparams.in_meta = True
    for attr_key in vars(args):
        if getattr(args, attr_key) is not None and attr_key in vars(hparams):
            setattr(hparams, attr_key, getattr(args, attr_key))
        # hparams.out_size = args.out_size
        # hparams.conv_size = args.conv_size
        # hparams.learning_rate = args.learning_rate
        # hparams.iters_per_checkpoint = args.iters_per_checkpoint
        # hparams.epochs = args.epochs

    # os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    if not hparams.data_parallel and not hparams.in_meta:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    
    output_directory = Path(args.output_directory) / convert_hparams_to_string(hparams)

    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    
    train(output_directory, args.log_directory, args.checkpoint_path, args.warm_start, hparams)
