import os
import time
import argparse
import pickle

from math import inf as mathinf
from json import dump as json_dump, load as json_load
import numpy as np
from tqdm import tqdm
from numpy import finfo
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from adamp import AdamP

from model import ContourEncoder
from melody_utils import MelodyDataset, MelodyCollate, MelodyPreSet, ContourSet, ContourCollate, pad_collate
from torch.optim.lr_scheduler import StepLR
from logger import AutoEncoderLogger
from hparams import create_hparams, HParams
from loss_function import SiameseLoss
from validation import get_contour_embeddings, cal_ndcg, cal_ndcg_single

from metalearner.common.config import experiment, worker
from metalearner.api import scalars


def prepare_dataloaders(hparams, valid_only=False):
    # Get data, data loaders and collate function ready

    trainset = ContourSet(hparams.contour_path, set_type='train', pre_load=True)
    entireset = ContourSet(hparams.contour_path, set_type='entire', pre_load=True, num_aug_samples=0, num_neg_samples=0)
    validset =  ContourSet(hparams.contour_path, set_type='valid', pre_load=True, num_aug_samples=4, num_neg_samples=0)

    train_loader = DataLoader(trainset, hparams.batch_size, shuffle=True,num_workers=hparams.num_workers,
        collate_fn=ContourCollate(hparams.num_pos_samples, hparams.num_neg_samples), pin_memory=True)
    entire_loader = DataLoader(entireset, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
        collate_fn=ContourCollate(0, 0), pin_memory=True, drop_last=False)
    valid_loader = DataLoader(validset, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
        collate_fn=ContourCollate(hparams.num_pos_samples, 0), pin_memory=True, drop_last=False)

    return train_loader, valid_loader, entire_loader #, comparison_loader, collate_fn
    # return train_loader, #valid_loader, list_collate_fn


def prepare_directories_and_logger(output_directory, log_directory,):
    print(output_directory, log_directory)
    logger = AutoEncoderLogger(output_directory / log_directory)
    return logger


def load_model(hparams):
    model = ContourEncoder(hparams).cuda()
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

def validate(model, val_loader, entire_loader, logger, epoch, iteration, criterion, hparams):
    """Handles all the validation scoring and printing"""
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        if not 'melody' in hparams.model_code:
            total_embs, total_song_ids = get_contour_embeddings(model, entire_loader)
        for j, batch in enumerate(tqdm(val_loader)):
            if 'melody' in hparams.model_code:
                batch = batch.cuda()
                if hparams.data_parallel:
                    pitch_decoded, duration_decoded, input_lengths = model.module.validate(batch)
                else:
                    pitch_decoded, duration_decoded, input_lengths = model.validate(batch)
                loss = sum([criterion(pitch_decoded[i:i+1,:,:input_lengths[i]], batch[i:i+1,:input_lengths[i],0 ]) 
                            for i in range(batch.shape[0]) ]) / batch.shape[0]
                loss += sum([criterion(duration_decoded[i:i+1,:,:input_lengths[i]], batch[i:i+1,:input_lengths[i],1])
                            for i in range(batch.shape[0]) ]) / batch.shape[0]
                valid_loss += loss.item()
            else:
                contours, song_ids = batch
                anchor = model(contours.cuda())
                anchor_norm = anchor / anchor.norm(dim=1)[:, None]
                similarity = torch.mm(anchor_norm, total_embs.transpose(0,1))
                recommends = torch.topk(similarity, k=hparams.num_recom, dim=-1)[1]
                recommends = total_song_ids[recommends]
                ndcg = [cal_ndcg_single(recommends[i,:], song_ids[i]) for i in range(recommends.shape[0])]
                ndcg = sum(ndcg) / len(ndcg)
                valid_loss += ndcg

        valid_loss = valid_loss/(j+1)
    model.train()
    print("Valdiation Loss {}: {:5f} ".format(iteration, valid_loss))
    # if 'siamese' in hparams.model_code:
    #     print("Validation loss {}: {:9f}  ".format(iteration, valid_ndcg))
    # else:
    #     print("Valdiation Score {}: {:5f} ".format(iteration, valid_ndcg))
    #     valid_ndcg = -valid_ndcg
    # audio_sample = valset.convert_spec_to_wav(y_pred.squeeze(0).cpu().numpy())
    if hparams.in_meta:
        results = {'validation score': valid_loss}
        response = scalars.send_valid_result(worker.id, epoch, iteration, results)
    else:
        logger.log_validation(valid_loss, model, iteration)

    return valid_loss

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
    if hparams.optimizer_type.lower() == 'adamp':
        optimizer = AdamP(model.parameters(), lr=learning_rate,
                                    weight_decay=hparams.weight_decay)
    # elif hparams.optimizer_type.lower() == 'adam':
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                    weight_decay=hparams.weight_decay)
    # elif hparams.optimizer_type.lower() == 'sgdp':
    #     optimizer = SGDP(model.parameters(), lr=learning_rate,
    #                                 weight_decay=hparams.weight_decay, momentum=hparams.momentum)
    # elif hparams.optimizer_type.lower() == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
    #                                 weight_decay=hparams.weight_decay, momentum=hparams.momentum)   


    # criterion = torch.nn.CrossEntropyLoss().to('cuda')
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
    if 'melody' in hparams.model_code:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = SiameseLoss(margin=0.5)
    best_valid_loss = mathinf
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for _, batch in enumerate(train_loader):
            start = time.perf_counter()
            model.zero_grad()
            batch = batch.cuda()

            if 'melody' in hparams.model_code:
                pitch_decoded, duration_decoded, input_lengths = model(batch)
                # batch = batch[sorted_idx]
                loss = sum([criterion(pitch_decoded[i:i+1,:,:input_lengths[i]], batch[i:i+1,:input_lengths[i],0 ]) for i in range(batch.shape[0]) ]) / batch.shape[0]
                # loss = criterion(pitch_decoded, batch[:,:,0])
                loss += sum([criterion(duration_decoded[i:i+1,:,:input_lengths[i]], batch[i:i+1,:input_lengths[i],1]) for i in range(batch.shape[0]) ]) / batch.shape[0]
                # loss += criterion(duration_decoded, batch[:,:,1])
            else:
                anchor, pos, neg = model.siamese(batch)
                loss = criterion(anchor, pos, neg)
            reduced_loss = loss.item()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()

            duration = time.perf_counter() - start
            # print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
            #     iteration, reduced_loss, grad_norm, duration))
            if hparams.in_meta:
                results = {'training loss': -reduced_loss}
                # response = scalars.send_train_result(worker.id, epoch, iteration, results)
            else:
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if iteration % hparams.iters_per_checkpoint == 0: # and not iteration==0:
                del loss, batch
                valid_loss = validate(model, val_loader, entire_loader, logger, epoch, iteration, criterion, hparams)
                is_best = valid_loss < best_valid_loss
                best_valid_loss = min(valid_loss, best_valid_loss)
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

            iteration += 1


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
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--device', type=int, default=0,
                    required=False, help='gpu device index')
    parser.add_argument('--in_metalearner', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether work in meta learner')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--conv_size', type=int, default=128)        
    parser.add_argument('--out_size', type=int, default=256)        
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=64)

    parser.add_argument('--epochs', type=int, default=100)    
    parser.add_argument('--iters_per_checkpoint', type=int, default=5000)        
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--drop_out', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=16)        
    parser.add_argument('--model_code', type=str, default="contour_ae")
    parser.add_argument('--optimizer_type', type=str, default="adam")
    parser.add_argument('--num_neg_samples', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--average_pool', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether use average pool instaed of max pool')
    parser.add_argument('--encoder_size', type=int, default=32)        
    parser.add_argument('--middle_size', type=int, default=16)        
    parser.add_argument('--latent_vec_size', type=int, default=10)

    args = parser.parse_args()
    if args.checkpoint_path:
        hparams = load_hparams(args.checkpoint_path)
    else:
        # hparams = create_hparams(args.hparams)
        hparams = HParams()
    
    
    if args.in_metalearner:
        hparams.in_meta = True
        for attr_key in vars(args):
            if attr_key in vars(hparams):
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
