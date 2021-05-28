import argparse

def create_parser():
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
    parser.add_argument('--data_parallel', type=lambda x: (str(x).lower() == 'true'), default=False, help='train with data parallel')

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
    parser.add_argument('--use_elementwise_loss', type=lambda x: (str(x).lower() == 'true'))

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

    return parser