import torch
import torch.nn as nn
import random

from torch.nn.modules.pooling import MaxPool1d

class ContourEncoder(nn.Module):
    def __init__(self, hparams):
        super(ContourEncoder, self).__init__()
        self.hidden_size = hparams.hidden_size
        self.input_size = hparams.input_size
        self.num_layers = hparams.num_layers
        self.embed_size = hparams.embed_size

        self.encoder = nn.LSTM(hparams.input_size, hparams.hidden_size, num_layers=hparams.num_layers, batch_first=True, dropout=hparams.drop_out, bidirectional=True)
        self.num_pos_samples = hparams.num_pos_samples
        self.num_neg_samples = hparams.num_neg_samples

        self.fc = nn.Linear(hparams.hidden_size * 2 * hparams.num_layers , hparams.embed_size)

    def forward(self, batch):
        _, hidden = self.encoder(batch)
        num_sequence = hidden[0].shape[1]
        out = hidden[0].permute(0,1,2).view(num_sequence, -1)
        return self.fc(out)

    def siamese(self, batch):
        out = self(batch)
        out = out.view([-1, 1+self.num_pos_samples+self.num_neg_samples, out.shape[1]])
        return out[:,0:1,:], out[:,1:1+self.num_pos_samples,:], out[:,1+self.num_pos_samples:,:]
        # seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # seq_unpacked = self.fc(seq_unpacked)
        # seq_unpacked = seq_unpacked.view([-1, 1+self.num_pos_samples+self.num_neg_samples, seq_unpacked.shape[1],seq_unpacked.shape[2]])
        # return seq_unpacked[:,0:1,:], seq_unpacked[:,1:1+self.num_pos_samples,:], seq_unpacked[:,1+self.num_pos_samples:,:]


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvNorm, self).__init__()
        self.conv_norm = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_norm(x)

class CnnEncoder(nn.Module):
    def __init__(self, hparams):
        super(CnnEncoder, self).__init__()
        self.hidden_size = hparams.hidden_size
        self.input_size = hparams.input_size
        self.conv_size = 3
        self.num_layers = hparams.num_layers
        self.embed_size = hparams.embed_size

        self.encoder = nn.Sequential(
            ConvNorm(2, self.hidden_size, self.conv_size, 0),
            nn.MaxPool1d(3),
            ConvNorm(self.hidden_size, self.hidden_size, self.conv_size, 0),
            nn.MaxPool1d(3),
            ConvNorm(self.hidden_size, self.hidden_size, self.conv_size, 0),
            nn.MaxPool1d(3),
            ConvNorm(self.hidden_size, self.hidden_size, self.conv_size, 0),
            
        )
        self.fc = nn.Linear(hparams.hidden_size, hparams.embed_size)

        self.num_pos_samples = hparams.num_pos_samples
        self.num_neg_samples = hparams.num_neg_samples

    def forward(self, batch):
        out = self.encoder(batch.permute(0,2,1))
        out = nn.functional.max_pool1d(out, out.shape[-1])

        return self.fc(out[:,:,0])

    def siamese(self, batch):
        out = self(batch)
        out = out.view([-1, 1+self.num_pos_samples+self.num_neg_samples, out.shape[1]])
        return out[:,0:1,:], out[:,1:1+self.num_pos_samples,:], out[:,1+self.num_pos_samples:,:]
        # seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # seq_unpacked = self.fc(seq_unpacked)
        # seq_unpacked = seq_unpacked.view([-1, 1+self.num_pos_samples+self.num_neg_samples, seq_unpacked.shape[1],seq_unpacked.shape[2]])
        # return seq_unpacked[:,0:1,:], seq_unpacked[:,1:1+self.num_pos_samples,:], seq_unpacked[:,1+self.num_pos_samples:,:]
        