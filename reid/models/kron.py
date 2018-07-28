import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def kron_matching(*inputs):
    assert len(inputs) == 2
    assert inputs[0].dim() == 4 and inputs[1].dim() == 4
    assert inputs[0].size() == inputs[1].size()
    N, C, H, W = inputs[0].size()

    # Convolve every feature vector from inputs[0] with inputs[1]
    #   In: x0, x1 = N x C x H x W
    #   Proc: weight = x0, permute to (NxHxW) x C x 1 x 1
    #         input = x1, view as 1 x (NxC) x H x W
    #   Out: out = F.conv2d(input, weight, groups=N)
    #            = 1 x (NxHxW) x H x W, view as N x H x W x (HxW)
    w = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, C, 1, 1)
    x = inputs[1].view(1, N * C, H, W)
    x = F.conv2d(x, w, groups=N)
    x = x.view(N, H, W, H, W)
    return x


class KronMatching(nn.Module):
    def __init__(self):
        super(KronMatching, self).__init__()

    def forward(self, *inputs):
        return kron_matching(*inputs)
