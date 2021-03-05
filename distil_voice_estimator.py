import os
import time
import argparse
import _pickle as pickle
import copy

from math import inf as mathinf
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from nnAudio import Spectrogram

# from adamp import AdamP

from model import Melody_ResNet
from data_utils import AudioOnlySet, get_song_ids_of_selected_genre, AudioSpecCollate
from torch.optim.lr_scheduler import StepLR
from logger import Logger
from hparams import HParams

from metalearner.common.config import experiment, worker
from metalearner.api import scalars

        
def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    with open('flo_metadata.dat', 'rb') as f:
        metadata = pickle.load(f)
    # selected_genres = [4, 12, 13, 17, 10, 7,15, 11, 9]
    selected_genres= [4]

    song_ids = get_song_ids_of_selected_genre(metadata, selected_genre=selected_genres)

    trainset = AudioOnlySet(hparams.data_dir,  song_ids, set_type='train')
    validset =  AudioOnlySet(hparams.data_dir, song_ids, set_type='valid')

    train_loader = DataLoader(trainset, hparams.batch_size, shuffle=True,num_workers=hparams.num_workers,
        collate_fn=AudioSpecCollate(), pin_memory=True)
    valid_loader = DataLoader(validset, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
        collate_fn=AudioSpecCollate(), pin_memory=True, drop_last=False)

    return train_loader, valid_loader #, comparison_loader, collate_fn
    # return train_loader, #valid_loader, list_collate_fn

def prepare_directories_and_logger(output_directory, log_directory,):
    print(output_directory, log_directory)
    logger = Logger(output_directory / log_directory)
    return logger

def load_model(hparams, checkpoint_path='/home/svcapp/userdata/dev/melodyExtraction_SSL/weights/torch_weights.pt'):
    hparams_b = copy.copy(hparams)
    hparams_b.input_size = 512
    model = Melody_ResNet()
    model.spec_layer = Spectrogram.STFT(n_fft=1024, freq_bins=None,
                                hop_length=800, window='hann',
                                freq_scale='no', center=True,
                                pad_mode='reflect', sr=8000, trainable=False,
                                output_format='Magnitude')
    model = model.to('cuda')
    teacher_model = Melody_ResNet().cuda()
    teacher_model.load_state_dict(torch.load(checkpoint_path))
    if hparams.data_parallel:
        model = torch.nn.DataParallel(model)
        teacher_model = torch.nn.DataParallel(teacher_model)
    return model, teacher_model

def load_checkpoint(checkpoint_path, model, optimizer, train_on_humming=False):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    if train_on_humming:
        return model, optimizer, 0, 0
    learning_rate = checkpoint_dict['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=optimizer.defaults['weight_decay'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
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
    with open(output_name, 'wb') as f:
        pickle.dump(hparams, f)

def load_hparams(checkpoint_path):
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    hparams_path = checkpoint_path.parent / 'hparams.dat'
    with open(hparams_path, 'rb') as f:
        return pickle.load(f)

def convert_hparams_to_string(hparams):
    out_string =  'voice_estimator_{}_{}/'.format(
        hparams.learning_rate, 
        datetime.now().strftime('%y%m%d-%H%M%S')
        )
    if hparams.in_meta:
        out_string = f"worker_{str(worker.id)}_{out_string}"
    return out_string


def validate(model, val_loader, entire_loader, logger, epoch, iteration, hparams, record_key="validation_score"):
    """Handles all the validation scoring and printing"""
    model.eval()
    valid_score = {}
    with torch.no_grad():
        total_embs, total_song_ids = get_contour_embeddings(model, entire_loader)
        valid_score[record_key] = cal_ndcg_of_loader(model, val_loader, total_embs, total_song_ids)
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
        print("Valdiation nDCG {}: {:5f} ".format(iteration, valid_score[record_key]))
    else:
        score_string = "Valdiation nDCG {}: {:5f} ".format(iteration, valid_score[record_key])
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

    return valid_score[record_key]

def train(output_directory, log_directory, checkpoint_path, hparams):
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
    model, teacher_model = load_model(hparams)
    learning_rate = hparams.learning_rate
    # if hparams.optimizer_type.lower() == 'adamp':
    #     optimizer = AdamP(model.parameters(), lr=learning_rate,
    #                                 weight_decay=hparams.weight_decay)
    # else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=hparams.weight_decay)

    logger = prepare_directories_and_logger(output_directory, log_directory)
    train_loader, val_loader = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        model, optimizer, _learning_rate, iteration = load_checkpoint(
            checkpoint_path, model, optimizer, args.train_on_humming)
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))
    else:
        save_hparams(hparams, output_directory)

    scheduler = StepLR(optimizer, step_size=hparams.learning_rate_decay_steps,
                       gamma=hparams.learning_rate_decay_rate)
    model.train()
    teacher_model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    best_valid_score = 0
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        # print("Epoch: {}".format(epoch))
        for _, batch in enumerate(train_loader):
            start = time.perf_counter()
            model.zero_grad()
            # batch = batch.cuda()
            audio, spec_10 = batch
            audio = audio.to('cuda')
            spec_10 = spec_10.to('cuda')
            spec_100 = model.spec_layer(audio)
            spec_100 = torch.log10(spec_100 + 1e-10)
            spec_100 = spec_100.permute(0,2,1).unsqueeze(1)
            train_result, _ = model(spec_100)
            with torch.no_grad():
                teacher_result, _ = teacher_model(spec_10)
            teacher_result = teacher_result.view(spec_100.shape[0], -1, teacher_result.shape[2]).view(spec_100.shape[0], -1, 10, teacher_result.shape[2])
            loss = criterion(train_result.permute(0,2,1), torch.mean(teacher_result, dim=2).permute(0,2,1))
            reduced_loss = loss.item()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()
            scheduler.step()
            duration = time.perf_counter() - start
            if hparams.in_meta:
                results = {'training loss': -reduced_loss}
                # response = scalars.send_train_result(worker.id, epoch, iteration, results)
            else:
                logger.log_training(
                    reduced_loss, grad_norm.item(), learning_rate, duration, iteration)


            if iteration % hparams.iters_per_checkpoint == 1: # and not iteration==0:
                valid_score = validate(model, val_loader, entire_loader, logger, epoch, iteration, hparams)
                fine_tune_model = model
                is_best = valid_score > best_valid_score
                best_valid_score = max(valid_score, best_valid_score)
                if is_best:
                    checkpoint_path = output_directory / 'checkpoint_best.pt'
                    save_checkpoint(fine_tune_model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
                else:
                    checkpoint_path = output_directory / 'checkpoint_last.pt'
                    save_checkpoint(fine_tune_model, optimizer, learning_rate, iteration,
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
    parser.add_argument('--humming_path', type=str,
                    help='path to contour.json')
    parser.add_argument('--data_dir', type=str,
                    help='path to pitch txt dir')
    parser.add_argument('--in_metalearner', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether work in meta learner')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--hidden_size', type=int, required=False)        
    parser.add_argument('--embed_size', type=int, required=False)
    parser.add_argument('--kernel_size', type=int, required=False)
    parser.add_argument('--compression_ratio', type=int, required=False)

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
    parser.add_argument('--num_pos_samples', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--loss_margin', type=float)
    parser.add_argument('--min_vocal_ratio', type=float)

    parser.add_argument('--summ_type', type=str)
    parser.add_argument('--use_pre_encoder', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--is_scheduled', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--get_valid_by_aug', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--use_res', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--use_gradual_size', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_on_humming', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--iters_per_humm_train', type=int)
    parser.add_argument('--combined_training', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--epoch_for_humm_train', type=int)

    parser.add_argument('--add_abs_noise', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--add_smoothing', type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--mask_w', type=float)
    parser.add_argument('--tempo_w', type=float)
    parser.add_argument('--tempo_slice', type=int)
    parser.add_argument('--drop_w', type=float)
    parser.add_argument('--std_w', type=float)
    parser.add_argument('--pitch_noise_w', type=float)
    parser.add_argument('--fill_w', type=float)
    parser.add_argument('--abs_noise_r', type=float)
    parser.add_argument('--abs_noise_w', type=float)


    args = parser.parse_args()
    if args.checkpoint_path:
        hparams = load_hparams(args.checkpoint_path)
        dummy = HParams()
        hparams.contour_path = dummy.contour_path
        hparams.humming_path = dummy.humming_path
        hparams.in_meta = False
        hparams.get_valid_by_aug = False

        # hparams.use_gradual_size = True
        # hparams.kernel_size = 5
        # hparams.embed_size = 256
        # hparams.summ_type = 'rnn'
        # hparams.combined_training=False
        # hparams.train_on_humming=True
        dummy_hparams = HParams()
        for key in vars(dummy_hparams):
            if not hasattr(hparams, key):
                setattr(hparams, key, getattr(dummy_hparams, key))
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

    
    train(output_directory, args.log_directory, args.checkpoint_path, hparams)
