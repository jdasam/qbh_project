class HParams:
    def __init__(self):
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs=5000
        self.iters_per_checkpoint=5000
        self.seed=1234
        self.dynamic_loss_scaling=True
        self.cudnn_enabled=True
        self.cudnn_benchmark=False
        self.data_parallel=False
        # ignore_layers=['embedding.weight']

        ################################
        # Data Parameters             #
        ################################
        self.flo_dir_path = "/home/svcapp/flo_ssd/"
        self.idx_dict_fname = "/home/svcapp/userdata/musicai/melon/index_dict.dat"
        self.mel_dir_path = "/home/svcapp/userdata/musicai/melon/arena_mel"
        self.pitch_path = '/home/svcapp/userdata/flo_melody/melody_subgenre.dat'
        # self.contour_path = '/home/svcapp/userdata/flo_melody/contour_subgenre_norm.json'
        self.contour_path = 'contour_tiny.json'

        ################################
        # Model Parameters             #
        ################################
        # Encoder parameters
        self.input_size = 2
        self.hidden_size = 128
        self.num_layers = 4 # conv layers
        self.kernel_size = 3
        self.embed_size = 128
        self.num_head = 8
        self.use_attention = False
        self.use_context_attention = False
        self.use_rnn = False
        self.use_pre_encoder = False
        self.use_res = False
        self.use_gradual_size = False

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate=False
        self.optimizer_type='adam'
        self.learning_rate=1e-4
        self.learning_rate_decay_steps = 30000
        self.learning_rate_decay_rate = 0.99
        self.weight_decay=1e-6
        self.momentum = 0.9
        self.center_loss_weight = 0.1
        self.num_recom = 50
        self.is_scheduled = False
        self.get_valid_by_aug = False

        self.grad_clip_thresh=1.0
        self.num_workers = 4
        self.batch_size = 24
        self.valid_batch_size = 64
        self.drop_out = 0.2
        self.loss_margin = 0.5
        self.model_code='contour_scheduled'
        self.pos_loss_weight = 1e4
        self.num_neg_samples = 4
        self.num_pos_samples = 2
        self.pre_load_mel = False
        self.in_meta = False

