import os
import time
import _pickle as pickle
import copy
import random

from math import inf as mathinf
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from model.logger import Logger
from model.hparams import HParams
from model.loss_function import SiameseLoss
from model.model import CnnEncoder, CombinedModel
from utils.data_utils import ContourCollate, HummingPairSet, WindowedContourSet, get_song_ids_of_selected_genre, AudioSet, AudioCollate, AudioContourCollate, HummingAudioSet
from model.validation import get_contour_embeddings, cal_mrr_of_loader
from utils.parser_util import create_parser
from model import hparams
import sys
sys.modules['hparams'] = hparams

def prepare_humming_db_loaders(hparams):
    with open(hparams.humming_path, "rb") as f:
        contour_pairs = pickle.load(f)
    aug_keys = ['tempo', 'key', 'std', 'pitch_noise']
    if hparams.add_abs_noise:
        aug_keys.append('absurd_noise')
    if hparams.add_smoothing:
        aug_keys.append('smoothing')
    aug_weights = make_aug_param_dictionary(hparams)



    if hparams.end_to_end:
        train_set = HummingAudioSet(hparams.data_dir, contour_pairs, aug_weights, "train", aug_keys, num_aug_samples=hparams.num_pos_samples, num_neg_samples=hparams.num_neg_samples)
        valid_set = HummingAudioSet(hparams.data_dir, contour_pairs, [], "valid", aug_keys, num_aug_samples=0, num_neg_samples=0)
        train_loader = DataLoader(train_set, hparams.batch_size, shuffle=True,num_workers=hparams.num_workers,
            collate_fn=AudioContourCollate(), pin_memory=True)
        valid_loader = DataLoader(valid_set, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
            collate_fn=ContourCollate(0, 0, for_cnn=True), pin_memory=True, drop_last=False)
    else:
        train_set = HummingPairSet(contour_pairs, aug_weights, "train", aug_keys, num_aug_samples=hparams.num_pos_samples, num_neg_samples=hparams.num_neg_samples)
        valid_set = HummingPairSet(contour_pairs, [], "valid", aug_keys, num_aug_samples=0, num_neg_samples=0)
        train_loader = DataLoader(train_set, hparams.batch_size, shuffle=True,num_workers=hparams.num_workers,
            collate_fn=ContourCollate(hparams.num_pos_samples, hparams.num_neg_samples, for_cnn=True), pin_memory=True)
        valid_loader = DataLoader(valid_set, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
            collate_fn=ContourCollate(0, 0, for_cnn=True), pin_memory=True, drop_last=False)
    return train_loader, valid_loader


def prepare_entire_loader(hparams, add_humm=True, num_song_limit=None):
    with open(hparams.meta_path, 'rb') as f:
        metadata = pickle.load(f)
    selected_genres = [4, 12, 13, 17, 10, 7,15, 11, 9]
    # selected_genres = [4]
    song_ids = get_song_ids_of_selected_genre(metadata, selected_genre=selected_genres)
    if num_song_limit and len(song_ids) > num_song_limit:
        song_ids = random.sample(song_ids, num_song_limit)
    if add_humm:
        with open('data/humm_db_ids.dat', 'rb') as f:
            humm_ids = pickle.load(f)
        song_ids += humm_ids

    if hparams.end_to_end:
        entire_set = AudioSet(hparams.data_dir, aug_weights=[], song_ids=song_ids, set_type='entire', num_aug_samples=0, num_neg_samples=0, min_vocal_ratio=hparams.min_vocal_ratio, sample_rate=hparams.sample_rate)
        entire_loader = DataLoader(entire_set, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
            collate_fn=AudioCollate(), pin_memory=True, drop_last=False)
    else:
        entire_set = WindowedContourSet(hparams.data_dir, aug_weights=[], song_ids=song_ids, set_type='entire', num_aug_samples=0, num_neg_samples=0, min_vocal_ratio=hparams.min_vocal_ratio)
        entire_loader = DataLoader(entire_set, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
            collate_fn=ContourCollate(0, 0, for_cnn=True), pin_memory=True, drop_last=False)
    return entire_loader
        
def prepare_dataloaders(hparams, entire_loader):
    # Get data, data loaders and collate function ready
    aug_weights = make_aug_param_dictionary(hparams)

    if hparams.end_to_end:
        trainset = AudioSet(entire_loader.dataset.path, aug_weights, set_type='train', pre_load_data=entire_loader.dataset.contours, num_aug_samples=hparams.num_pos_samples, num_neg_samples=hparams.num_neg_samples, sample_rate=hparams.sample_rate)
        validset =  AudioSet(entire_loader.dataset.path, aug_weights, set_type='valid', pre_load_data=entire_loader.dataset.contours, num_aug_samples=4, num_neg_samples=0, sample_rate=hparams.sample_rate)

        train_loader = DataLoader(trainset, hparams.batch_size, shuffle=True,num_workers=hparams.num_workers,
            collate_fn=AudioContourCollate(), pin_memory=True)
        valid_loader = DataLoader(validset, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
            collate_fn=ContourCollate(hparams.num_pos_samples, 0, for_cnn=True), pin_memory=True, drop_last=False)

    else:
        trainset = WindowedContourSet(entire_loader.dataset.path, aug_weights, set_type='train', pre_load_data=entire_loader.dataset.contours, num_aug_samples=hparams.num_pos_samples, num_neg_samples=hparams.num_neg_samples)
        validset =  WindowedContourSet(entire_loader.dataset.path, aug_weights, set_type='valid', pre_load_data=entire_loader.dataset.contours, num_aug_samples=4, num_neg_samples=0)
        train_loader = DataLoader(trainset, hparams.batch_size, shuffle=True,num_workers=hparams.num_workers,
            collate_fn=ContourCollate(hparams.num_pos_samples, hparams.num_neg_samples, for_cnn=True), pin_memory=True)
        valid_loader = DataLoader(validset, hparams.valid_batch_size, shuffle=False,num_workers=hparams.num_workers,
            collate_fn=ContourCollate(hparams.num_pos_samples, 0, for_cnn=True), pin_memory=True, drop_last=False)

    return train_loader, valid_loader #, comparison_loader, collate_fn

def make_aug_param_dictionary(hparams):
    aug_weight_keys = ['mask_w', 'tempo_w', 'tempo_slice', 'drop_w', 'std_w', 'pitch_noise_w', 'fill_w']
    return {key: getattr(hparams, key) for key in aug_weight_keys}

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

def load_end_to_end_model(hparams, checkpoint_path, voice_ckpt_path):
    hparams_b = copy.copy(hparams)
    hparams_b.input_size = 512
    model = CombinedModel(hparams, hparams_b)
    model.singing_voice_estimator.load_state_dict(torch.load(voice_ckpt_path)['state_dict'])
    model.contour_encoder.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    if hparams.data_parallel:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    return model

def load_checkpoint(checkpoint_path, model, optimizer, train_on_humming=False):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cuda')
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
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': state_dict,
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

def freeze_model(model):
    if hasattr(model, 'module'):
        if hasattr(model.module, 'freeze_except_audio_encoder'):
            model.module.freeze_except_audio_encoder()
    else:
        if hasattr(model, 'freeze_except_audio_encoder'):
            model.freeze_except_audio_encoder()

def unfreeze_model(model):
    if hasattr(model, 'module'):
        if hasattr(model.module, 'unfreeze_parameters'):
            model.module.unfreeze_parameters()
    else:
        if hasattr(model, 'unfreeze_parameters'):
            model.unfreeze_parameters()

def convert_hparams_to_string(hparams):
    out_string =  '{}_hidden{}_lr{}_{}/'.format(
        hparams.model_code, 
        hparams.hidden_size, 
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
        valid_score[record_key] = cal_mrr_of_loader(model, val_loader, total_embs, total_song_ids)
        # for j, batch in enumerate(tqdm(val_loader)):
        if hparams.get_valid_by_aug:
            aug_keys = [x for x in val_loader.dataset.aug_keys]
            for key in aug_keys:
                val_loader.dataset.aug_keys = [key]
                valid_score_of_key = cal_mrr_of_loader(model, val_loader, total_embs, total_song_ids)
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
    if hparams.in_meta:
        response = scalars.send_valid_result(worker.id, epoch, iteration, valid_score)
    else:
        logger.log_validation(valid_score, model, iteration)

    return valid_score[record_key]

def train(output_directory, log_directory, checkpoint_path, voice_ckpt_path, hparams):
    """Training and validation logging results to tensorboard

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    voice_ckpt_path(string): checkpoint path of pre-trained voice estimator model (for end-to-end model)
    hparams (object): hyperparameter object (defined in hparams.py)
    """

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    if hparams.end_to_end:
        model = load_end_to_end_model(hparams, checkpoint_path, voice_ckpt_path)
    else:
        model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=hparams.weight_decay)

    logger = prepare_directories_and_logger(output_directory, log_directory)
    entire_loader = prepare_entire_loader(hparams)

    # define dataloader based on hparams options
    if hparams.train_on_humming:
        train_loader, val_loader = prepare_humming_db_loaders(hparams)
    else:
        train_loader, val_loader = prepare_dataloaders(hparams, entire_loader)
    if hparams.combined_training:
        humm_train_loader, humm_val_loader = prepare_humming_db_loaders(hparams)


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
    freeze_model(model)
    criterion = SiameseLoss(margin=hparams.loss_margin, use_euclid=hparams.use_euclid, use_elementwise=hparams.use_elementwise_loss)
    best_valid_score = 0
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        # print("Epoch: {}".format(epoch))
        for _, batch in enumerate(train_loader):
            start = time.perf_counter()
            model.zero_grad()
            if hparams.end_to_end:
                batch = (x.cuda() for x in batch)
                anchor, pos, neg = model(batch, siamese=True, num_pos=train_loader.dataset.num_aug_samples)
            else:
                batch = batch.cuda()
                anchor, pos, neg = model(batch, siamese=True)
            loss = criterion(anchor, pos, neg)
            reduced_loss = loss.item()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()
            scheduler.step()
            duration = time.perf_counter() - start
            # if hparams.in_meta:
            #     results = {'training loss': -reduced_loss}
            #     # response = scalars.send_train_result(worker.id, epoch, iteration, results)
            # else:
            logger.log_training(
                reduced_loss, grad_norm, learning_rate, duration, iteration)

            if iteration % hparams.iters_per_checkpoint == 1: # and not iteration==0:
                if hparams.combined_training:
                    temp_check_path = output_directory / 'model_temp.pt'
                    save_checkpoint(model, optimizer, learning_rate, iteration, temp_check_path)
                    model = model.to('cpu')
                    fine_learning_rate = hparams.learning_rate
                    fine_tune_model = copy.deepcopy(model)
                    fine_tune_model = fine_tune_model.to('cuda')
                    fine_optimizer = torch.optim.Adam(fine_tune_model.parameters(), lr=fine_learning_rate,
                                weight_decay=hparams.weight_decay)
                    fine_tune_model.train()
                    unfreeze_model(fine_tune_model)
                    for fine_epoch in range(hparams.epoch_for_humm_train):
                        for batch in humm_train_loader:
                            fine_tune_model.zero_grad()
                            if hparams.end_to_end:
                                batch = (x.cuda() for x in batch)
                                anchor, pos, neg = fine_tune_model(batch, siamese=True, num_pos=train_loader.dataset.num_aug_samples)
                            else:
                                batch = batch.cuda()
                                anchor, pos, neg = fine_tune_model(batch, siamese=True)
                            fine_loss = criterion(anchor, pos, neg)
                            fine_loss.backward()
                            torch.nn.utils.clip_grad_norm_(fine_tune_model.parameters(), hparams.grad_clip_thresh)
                            fine_optimizer.step()
                    valid_score = validate(fine_tune_model, humm_val_loader, entire_loader, logger, epoch, iteration, hparams, record_key='humm_validation_score')
                    model = model.to('cuda')
                    model, optimizer, learning_rate, iteration = load_checkpoint(temp_check_path, model, optimizer)
                    freeze_model(model)
                    fine_tune_model = fine_tune_model.to('cpu')
                    orig_valid_score = validate(model, val_loader, entire_loader, logger, epoch, iteration, hparams, record_key='orig_validation_score')
                    if hparams.in_meta:
                        response = scalars.send_valid_result(worker.id, epoch, iteration, {'humm_validation_score': valid_score, 'orig_validation_score': orig_valid_score})
                else:
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
            iteration += 1


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    if args.checkpoint_path:
        hparams = load_hparams(args.checkpoint_path)
        dummy = HParams()
        hparams.humming_path = dummy.humming_path
        hparams.in_meta = False
        hparams.get_valid_by_aug = False
        dummy_hparams = HParams()
        for key in vars(dummy_hparams):
            if not hasattr(hparams, key):
                setattr(hparams, key, getattr(dummy_hparams, key))
    else:
        # hparams = create_hparams(args.hparams)
        hparams = HParams()
    
    if args.in_metalearner:
        from metalearner.common.config import experiment, worker
        from metalearner.api import scalars
        hparams.in_meta = True
    for attr_key in vars(args):
        if getattr(args, attr_key) is not None and attr_key in vars(hparams):
            setattr(hparams, attr_key, getattr(args, attr_key))

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    if not hparams.data_parallel and not hparams.in_meta:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
    
    output_directory = Path(args.output_directory) / convert_hparams_to_string(hparams)

    print("cuDNN Enabled:", hparams.cudnn_enabled)
    
    train(output_directory, args.log_directory, args.checkpoint_path, args.voice_ckpt_path, hparams)
