import torch
import torch.nn as nn
import torch.nn.functional as F  # Functional API (not heavily used here)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SmallTCN(nn.Module):
    def __init__(self, num_inputs, num_channels=[16, 32], num_classes=4, kernel_size=5, dropout=0.3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation_size,
                              padding=(kernel_size-1)*dilation_size,
                              dropout=dropout)
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Conv1d(num_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):  # x: (B,1,L)
        out = self.network(x)
        out = self.fc(out)  # (B,C,L)
        return out.transpose(1,2)  # (B,L,C)

from torchview import draw_graph  # For visualizing the model graph
import torch
model = SmallTCN(num_inputs=1, num_channels=[16,32], num_classes=4, kernel_size=5, dropout=0.3)
x = torch.randn(1, 1, 100)  # Dummy input used for inspection / visualization only

# expand_nested=False keeps it blockwise (Sequential, TemporalBlock, etc.)
from torchsummary import summary  # Summarize model architecture in the console
summary(model, input_size=(1, 100))
