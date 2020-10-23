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
        # ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        self.flo_dir_path = "/home/svcapp/flo_ssd/"
        self.idx_dict_fname = "/home/svcapp/userdata/musicai/melon/index_dict.dat"
        self.mel_dir_path = "/home/svcapp/userdata/musicai/melon/arena_mel"
        self.pitch_path = '/home/svcapp/userdata/flo_melody/melody_subgenre.dat'
        self.contour_path = '/home/svcapp/userdata/flo_melody/contour_subgenre_norm.json'
        # self.contour_path = 'contour_tiny.json'

        ################################
        # Model Parameters             #
        ################################
        # Encoder parameters
        self.input_size = 2
        self.hidden_size = 128
        self.num_layers = 2 # LSTM num layer
        self.embed_size = 128

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate=False
        self.optimizer_type='adam'
        self.learning_rate=1e-4
        self.learning_rate_decay_steps = 10000
        self.learning_rate_decay_rate = 0.98
        self.weight_decay=1e-6
        self.momentum = 0.9
        self.center_loss_weight = 0.1
        self.num_recom = 50

        self.grad_clip_thresh=1.0
        self.num_workers = 0
        self.batch_size = 32
        self.valid_batch_size = 128
        self.drop_out = 0.2
        self.model_code='contour'
        self.pos_loss_weight = 1e4
        self.num_neg_samples = 4
        self.num_pos_samples = 2
        self.pre_load_mel = False
        self.in_meta = False

