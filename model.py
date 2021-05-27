import torch
import torch.nn as nn
import math
import random
from nnAudio import Spectrogram


from torch.nn.modules.pooling import MaxPool1d
from module import ConvNorm, Res_1d
from model_utils import cal_conv_parameters

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


class CnnEncoder(nn.Module):
    def __init__(self, hparams):
        super(CnnEncoder, self).__init__()
        self.hidden_size = hparams.hidden_size
        self.input_size = hparams.input_size
        self.kernel_size = hparams.kernel_size
        self.num_layers = hparams.num_layers
        self.embed_size = hparams.embed_size

        if hparams.use_pre_encoder:
            self.use_pre_encoder = True
            self.pre_encoder = nn.Linear(self.input_size, self.hidden_size)
            self.cnn_input_size = self.hidden_size
        else:
            self.use_pre_encoder = False
            self.cnn_input_size = self.input_size
    
        if hparams.use_res:
            module = Res_1d 
        else:
            module = ConvNorm
        self.encoder = nn.Sequential()
        parameters = cal_conv_parameters(hparams, self.cnn_input_size)

        for i, param in enumerate(parameters):
            self.encoder.add_module(f'conv_{i}', module(param['input_channel'], param['output_channel'], self.kernel_size, self.kernel_size//2))
            if param['max_pool'] > 1:
                self.encoder.add_module(f'pool_{i}', nn.MaxPool1d(param['max_pool']))
        self.fc = nn.Linear(parameters[-1]['output_channel'], hparams.embed_size)

        self.num_pos_samples = hparams.num_pos_samples
        self.num_neg_samples = hparams.num_neg_samples

        if hparams.summ_type == "context_attention":
            self.final_attention = ContextAttention(parameters[-1]['output_channel'], num_head=hparams.num_head)
        elif hparams.summ_type == "simple_attention":
            self.final_attention = SimpleAttention(parameters[-1]['output_channel'])
        elif hparams.summ_type == "rnn":
            self.final_rnn = nn.GRU(input_size=parameters[-1]['output_channel'], hidden_size=parameters[-1]['output_channel'], num_layers=1, batch_first=True)
        self.summ_type = hparams.summ_type

    def forward(self, batch, siamese=False):
        if self.use_pre_encoder:
            batch = self.pre_encoder(batch)
        out = self.encoder(batch.permute(0,2,1))
        if self.summ_type == "context_attention" or self.summ_type == "simple_attention":
            out = self.final_attention(out.permute(0,2,1))
            out = self.fc(out)
        elif self.summ_type == "rnn":
            _, out = self.final_rnn(out.permute(0,2,1))
            out =  self.fc(out[0])
        else:
            out = nn.functional.max_pool1d(out, out.shape[-1])
            out = self.fc(out[:,:,0])
        if siamese:
            out = out.view([-1, 1+self.num_pos_samples+self.num_neg_samples, self.embed_size])
            return out[:,0:1,:], out[:,1:1+self.num_pos_samples,:], out[:,1+self.num_pos_samples:,:]
        else:
            return out

class SimpleAttention(nn.Module):
    def __init__(self, size):
        super(SimpleAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)

    def forward(self, x):
        attention = self.attention_net(x)
        softmax_weight = torch.softmax(attention, dim=1)
        attention_sum = softmax_weight * x
        return torch.sum(attention_sum, dim=1)



class ContextAttention(nn.Module):
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)
        self.head_size = int(size/num_head)
        self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        # self.context_vector = torch.nn.Parameter(torch.Tensor(size))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def get_attention(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        similarity = torch.bmm(attention_split, self.context_vector)
        return similarity

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)

        attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        similarity = torch.bmm(attention_split, self.context_vector.repeat(1, x.shape[0], 1).view(-1, self.head_size ,1))
        softmax_weight = torch.softmax(similarity, dim=1)
        x_split = torch.cat(x.split(split_size=self.head_size, dim=2), dim=0)
        weighted_mul = torch.bmm(softmax_weight.transpose(1,2), x_split)
        restore_size = int(weighted_mul.size(0) / self.num_head)
        attention = torch.cat(weighted_mul.split(split_size=restore_size, dim=0), dim=2)

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention



class ConvNormForRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvNormForRes, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        self.activation = nn.LeakyReLU(0.01)
    
    def forward(self, x, return_shortcut=False, skip_activation=False):
        shortcut = self.bn(self.conv(x))
        if return_shortcut:
            return self.activation(shortcut), shortcut
        elif skip_activation:
            return shortcut
        else:
            return self.activation(shortcut)
      

class ResNet_Block(nn.Module):
    def __init__(self, num_input_ch, num_channels):
        super(ResNet_Block, self).__init__()
        self.conv1 = ConvNormForRes(num_input_ch, num_channels, 1)
        self.conv2 = ConvNormForRes(num_channels, num_channels, 3)
        self.conv3 = ConvNormForRes(num_channels, num_channels, 3)
        self.conv4 = ConvNormForRes(num_channels, num_channels, 1)
        
    def cal_conv(self,x):
        return self.conv4(self.conv3(self.conv2(self.conv1(x))))

    def forward(self, x):
        x, shortcut = self.conv1(x, return_shortcut=True)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x, skip_activation=True)

        x += shortcut
        x = self.conv4.activation(x)
        x = torch.max_pool2d(x, (1,4))

        return x

class Melody_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            ResNet_Block(1, 64),
            ResNet_Block(64, 128),
            ResNet_Block(128, 192),
            ResNet_Block(192, 256),
        )

        # Keras uses a hard_sigmoid for default activation, but the test showed that using a plain sigmoid in PyTorch 
        # showed the most similar result with the pre-trained Keras Model
        # Also, PyTorch LSTM does not provides recurernt_dropout.
        self.lstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True,  dropout=0.3)
        num_output = int(55 * 2 ** (math.log(8, 2)) + 2)
        self.final = nn.Linear(512,num_output)

    def forward(self, input):
        block = self.block(input) # channel first for torch
        numOutput_P = block.shape[1] * block.shape[3]
        reshape_out = block.permute(0,2,3,1).reshape(block.shape[0], 31, numOutput_P)
        lstm_out, _ = self.lstm(reshape_out)
        return lstm_out

    def batch_slice_and_forward(self, input, batch_size=300):
        total_output = []
        for i in range(math.ceil(input.shape[0]/batch_size)):
            block = self.block(input[i*batch_size:(i+1)*batch_size]) # channel first for torch
            numOutput_P = block.shape[1] * block.shape[3]
            reshape_out = block.permute(0,2,3,1).reshape(block.shape[0], 31, numOutput_P)

            lstm_out, _ = self.lstm(reshape_out)
            out = self.final(lstm_out)
            out = torch.softmax(out, dim=-1)
            total_output.append(out)
        return torch.cat(total_output)


class Melody_ResNet_with_Spec(Melody_ResNet):
    def __init__(self):
        super(Melody_ResNet_with_Spec,self).__init__()
        self.spec_layer = Spectrogram.STFT(n_fft=1024, freq_bins=None,
                                hop_length=800, window='hann',
                                freq_scale='no', center=True,
                                pad_mode='reflect', sr=8000, trainable=False,
                                output_format='Magnitude')
    
    def forward(self, input):
        spec = self.spec_layer(input)
        spec = torch.log10(spec + 1e-10)
        spec = spec.permute(0,2,1).unsqueeze(1)
        block = self.block(spec) # channel first for torch
        numOutput_P = block.shape[1] * block.shape[3]
        # if block.shape[2] % 31 != 0:
        #     dummy = torch.zeros(block.shape[0], block.shape[1], (block.shape[2]//31+1)*31, block.shape[3])
        #     dummy[:,:,:block.shape[2]] = block 
        #     block = dummy
        reshape_out = block.permute(0,2,3,1).reshape(block.shape[0], -1, numOutput_P)
        lstm_out, _ = self.lstm(reshape_out)
        melody_out = self.final(lstm_out)
        melody_out = torch.softmax(melody_out, dim=-1)
        return  melody_out, lstm_out


class CombinedModel(nn.Module):
    def __init__(self, hparams, hparams_b):
        super().__init__()
        self.contour_encoder = CnnEncoder(hparams)
        self.singing_voice_estimator = Melody_ResNet_with_Spec()
        self.audio_encoder = CnnEncoder(hparams_b)
        self.embed_size = hparams.embed_size

    def freeze_except_audio_encoder(self):
        for param in self.singing_voice_estimator.parameters():
            param.requires_grad = False
        for param in self.contour_encoder.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input, num_pos=0, siamese=False):
        if siamese:
            audio_input, contour_pos_n_neg = input
            num_batch = audio_input.shape[0]
            _, voice_hidden = self.singing_voice_estimator(audio_input)
            audio_embedding = self.audio_encoder(voice_hidden)

            contour_embedding = self.contour_encoder(contour_pos_n_neg)
            contour_embedding = contour_embedding.view(num_batch, -1 , contour_embedding.shape[-1])

            return audio_embedding, contour_embedding[:,:num_pos], contour_embedding[:,num_pos:]
        elif input.shape[-1] == 2: # input is N x 2 contour :
            return self.contour_encoder(input)
        else:
            audio_input = input
            _, voice_hidden = self.singing_voice_estimator(audio_input)
            audio_embedding = self.audio_encoder(voice_hidden)
            return audio_embedding