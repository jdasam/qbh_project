from torch.utils.tensorboard import SummaryWriter

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)
    

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)


    def log_validation(self, reduced_loss, model, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
