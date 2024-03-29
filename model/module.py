import torch.nn as nn


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv_norm = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_norm(x)


class Res_1d(nn.Module):
    def __init__(self, input_channels, output_channels, shape, padding):
        super().__init__()
        # convolution
        self.conv_1 = nn.Conv1d(input_channels, output_channels, shape, padding=padding)
        self.bn_1 = nn.BatchNorm1d(output_channels)
        self.conv_2 = nn.Conv1d(output_channels, output_channels, shape, padding=padding)
        self.bn_2 = nn.BatchNorm1d(output_channels)

        # residual
        self.diff = False
        if input_channels != output_channels:
            self.conv_3 = nn.Conv1d(input_channels, output_channels, shape, padding=padding)
            self.bn_3 = nn.BatchNorm1d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out
