from torch import nn
import torch.nn.functional as F


class CTC(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super().__init__()
        self.n_class = n_class
        self.fc = nn.Linear(in_channels, n_class)

    def forward(self, x):  # [batch, 141, 512]
        x = self.fc(x)  # [batch, 141, 6773]
        x = F.log_softmax(x, dim=2)
        return x
