import math
import copy
from torch import nn
import torch
import torch.nn.functional as F
from .kron import KronMatching


class RandomWalkEmbed(nn.Module):
    def __init__(self, instances_num=4, feat_num=2048, num_classes=0, drop_ratio=0.5):
        super(RandomWalkEmbed, self).__init__()
        self.instances_num = instances_num
        self.feat_num = feat_num
        self.temp = 1
        self.kron = KronMatching()
        self.bn = nn.BatchNorm1d(feat_num)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        self.classifier = nn.Linear(feat_num, num_classes)
        self.classifier.weight.data.normal_(0, 0.001)
        self.classifier.bias.data.zero_()
        self.drop = nn.Dropout(drop_ratio)
        
    def _kron_matching(self, x1, x2):
        n, c, h, w = x1.size()
        x2_kro = self.kron(x1 / x1.norm(2, 1, keepdim=True).expand_as(x1),
                           x2 / x2.norm(2, 1, keepdim=True).expand_as(x2))
        x2_kro_att = F.softmax((self.temp * x2_kro).view(n * h * w, h * w), dim=1).view(n, h, w, h, w)
        warped_x2 = torch.bmm(x2.view(n, c, h * w), x2_kro_att.view(n, h * w, h * w).transpose(1, 2)).view(n, c, h, w)
        return warped_x2

    def forward(self, probe_x, gallery_x, p2g=True, g2g=False):
        if not self.training and len(probe_x.size()) != len(gallery_x.size()):
            probe_x = probe_x.unsqueeze(0)

        probe_x.contiguous()
        gallery_x.contiguous()

        if p2g is True:
            N_probe, C, H, W = probe_x.size()
            N_gallery = gallery_x.size(0)
            probe_x = probe_x.unsqueeze(1)
            probe_x = probe_x.expand(N_probe, N_gallery, C, H, W)
            probe_x = probe_x.contiguous()
            gallery_x = gallery_x.unsqueeze(0)
            gallery_x = gallery_x.expand(N_probe, N_gallery, C, H, W)
            gallery_x = gallery_x.contiguous()
            probe_x = probe_x.view(N_probe * N_gallery, C, H, W)
            gallery_x = gallery_x.view(N_probe * N_gallery, C, H, W)
            probe_x = self._kron_matching(gallery_x, probe_x)
            diff = F.avg_pool2d((probe_x - gallery_x), (probe_x - gallery_x).size()[2:])
        elif g2g is True:
            N_probe = probe_x.size(0)
            N_gallery = gallery_x.size(0)
            probe_x = F.avg_pool2d(probe_x, probe_x.size()[2:]).view(N_probe, self.feat_num)
            gallery_x = F.avg_pool2d(gallery_x, gallery_x.size()[2:]).view(N_gallery, self.feat_num)
            probe_x = probe_x.unsqueeze(1)
            probe_x = probe_x.expand(N_probe, N_gallery, self.feat_num)
            probe_x = probe_x.contiguous()
            gallery_x = gallery_x.unsqueeze(0)
            gallery_x = gallery_x.expand(N_probe, N_gallery, self.feat_num)
            gallery_x = gallery_x.contiguous()
            diff = gallery_x - probe_x

        diff = torch.pow(diff, 2)
        diff = diff.view(N_probe * N_gallery, -1)
        diff = diff.contiguous()
        bn_diff = self.bn(diff)
        bn_diff = self.drop(bn_diff)

        cls_encode = self.classifier(bn_diff)
        cls_encode = cls_encode.view(N_probe, N_gallery, -1)

        return cls_encode





class EltwiseSubEmbed(nn.Module):
    def __init__(self, nonlinearity='square', use_batch_norm=False,
                 use_classifier=False, num_features=0, num_classes=0):
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError("Unknown nonlinearity:", nonlinearity)
        self.use_batch_norm = use_batch_norm
        self.use_classifier = use_classifier
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.use_classifier:
            assert num_features > 0 and num_classes > 0
            self.classifier = nn.Linear(num_features, num_classes)
            self.classifier.weight.data.normal_(0, 0.001)
            self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_classifier:
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
        else:
            x = x.sum(1)
        return x



