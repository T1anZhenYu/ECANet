import torch
from torch import nn
from torch.nn.parameter import Parameter
import math
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        num_features: Number of num_featuress of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, num_features, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = num_features
        t = int(abs((math.log(num_features, 2) + 1) / 2))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class eca_layer_batchwise(nn.Module):
    """Constructs a ECA module.

    Args:
        num_features: Number of num_featuress of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, num_features, k_size=3,momentum=0.9):
        super(eca_layer_batchwise, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.register_buffer('running_mean', torch.zeros(num_features))
        t = int(abs((math.log(num_features, 2) + 1) / 2))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.momentum = momentum
    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        if self.training:
            mean = x.mean(dim=(0, 2, 3)).detach()
            # feature descriptor on the global spatial information
            # y = self.avg_pool(x)
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
            # Two different branches of ECA module
            y = self.conv(mean[None,None,:]).transpose(-1, -2).unsqueeze(-1)
        else:
            y = self.conv(self.running_mean[None,None,:]).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class NewBN(nn.Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        super(NewBN, self).__init__()

        t = int(abs((math.log(num_features, 2) + 1) / 2))
        k_size = t if t % 2 else t + 1

        self.linearvar = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.linearmean = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        n = x.numel() / (x.size(0))
        if self.training:

            mean = x.mean(dim=(0, 2, 3)).detach()
            var = (x-mean[None, :, None, None]).pow(2).mean(dim=(0,2, 3)).detach()

            # indexvar = torch.sqrt(self.bn.running_var).mean()/math.pow(n,0.5)
            # indexmean = self.bn.running_mean.mean() / math.pow(n, 0.5)

            meanmix = mean
            meanmix = self.sigmoid(self.linearmean(meanmix[None, None, :]).squeeze())

            varmix = torch.sqrt(var)
            varmix = self.sigmoid(self.linearvar(varmix[None,None,:]).squeeze())

            index = meanmix*0.5 + varmix*0.5

        else:
            mean = self.bn.running_mean
            var = self.bn.running_var

            # indexvar = torch.sqrt(self.bn.running_var).mean()/math.pow(n,0.5)
            # indexmean = self.bn.running_mean.mean() / math.pow(n, 0.5)

            meanmix = mean
            meanmix = self.sigmoid(self.linearmean(meanmix[None, None, :]).squeeze())

            varmix = torch.sqrt(var)
            varmix = self.sigmoid(self.linearvar(varmix[None,None,:]).squeeze())

            index = meanmix*0.5 + varmix*0.5

        out = self.bn(x)
        out.mul_(index[None, :, None, None])
        return out

