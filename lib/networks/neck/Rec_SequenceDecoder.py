import torch
import torch.nn.functional as F
from torch import nn


class DecoderWithRNN(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        rnn_hidden_size = kwargs.get('hidden_size', 96)
        self.out_channels = rnn_hidden_size * 2
        self.layers = 2
        self.lstm = nn.LSTM(in_channels, rnn_hidden_size, bidirectional=True, batch_first=True, num_layers=self.layers)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        return x


class Reshape(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape  # batch * 512 * 1 * 141
        assert H == 1
        x = x.reshape(B, C, H * W)  # batch * 512 * 141
        x = x.permute((0, 2, 1))  # (NTC)(batch, width, channels)
        # batch * 141 * 512
        # LSTM input: [batch, seq_len, input_size]
        return x


class SequenceDecoder(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.reshape = Reshape(in_channels)
        self.decoder = DecoderWithRNN(in_channels, **kwargs)
        self.out_channels = self.decoder.out_channels

    def forward(self, x):  # batch * 512 * 1 * 141
        x = self.reshape(x)  # batch * 141 * 512
        x = self.decoder(x)  # batch * 141 * 192
        return x


class BidiretionalLSTM(nn.Module):
    """After CNN backbone
    Args：
        nIn:(int)
    """
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class CRNN_neck(nn.Module):
    def __init__(self, nh, nclass):
        super().__init__()
        self.rnn = nn.Sequential(
            BidiretionalLSTM(512, nh, nh),
            BidiretionalLSTM(nh, nh, nclass))

    def forward(self, x):
        b, c, h, w = x.size()  # batch * 512 * 1 * 141
        assert h == 1, 'the height of neck input must be 1'
        x = x.squeeze(2)  # batch * 512 * 65
        x = x.permute(2, 0, 1)  # 65 * batch * 512  [width, batch_size, channel]
        output = F.log_softmax(self.rnn(x), dim=2)  # 65 * batch * 37
        return output.permute(1, 0, 2)