from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


def random_walk_compute(p_g_score, g_g_score, alpha):
    # Random Walk Computation
    one_diag = Variable(torch.eye(g_g_score.size(0)).cuda(), requires_grad=False)
    g_g_score_sm = Variable(g_g_score.data.clone(), requires_grad=False)
    # Row Normalization
    inf_diag = torch.diag(torch.Tensor([-float('Inf')]).expand(g_g_score.size(0))).cuda() + g_g_score_sm[:, :,1].squeeze().data
    A = F.softmax(Variable(inf_diag))
    A = (1 - alpha) * torch.inverse(one_diag - alpha * A)
    A = A.transpose(0, 1)
    p_g_score = torch.matmul(p_g_score.permute(2, 0, 1), A).permute(1, 2, 0).contiguous()
    g_g_score = torch.matmul(g_g_score.permute(2, 0, 1), A).permute(1, 2, 0).contiguous()
    p_g_score = p_g_score.view(-1, 2)
    g_g_score = g_g_score.view(-1, 2)
    outputs = torch.cat((p_g_score, g_g_score), 0)
    outputs = outputs.contiguous()

    return outputs


class RandomWalkKpmNet(nn.Module):
    def __init__(self, instances_num=4, base_model=None, embed_model=None, alpha=0.1):
        super(RandomWalkKpmNet, self).__init__()
        self.instances_num = instances_num
        self.alpha = alpha
        self.base = base_model
        self.embed = embed_model
        for i in range(len(embed_model)):
            setattr(self, 'embed_'+str(i), embed_model[i])

    def forward(self, x):
        x = self.base(x)
        N, C, H, W = x.size()
        probe_num = int(N / self.instances_num)
        gallery_num = int(N - N / (self.instances_num))
        x = x.view(probe_num, self.instances_num, C, H, W)

        probe_x = x[:, 0, :, :, :]
        probe_x = probe_x.contiguous()
        probe_x = probe_x.view(probe_num, C, H, W)
        gallery_x = x[:, 1:self.instances_num, :, :, :]
        gallery_x = gallery_x.contiguous()
        gallery_x = gallery_x.view(gallery_num, C, H, W)

        count = 2048 / (len(self.embed))
        outputs = []
        for j in range(len(self.embed)):
            for i in range(len(self.embed)):
                p_g_score = self.embed[j](probe_x[:,i*count:(i+1)*count].contiguous(),
                                          gallery_x[:,i*count:(i+1)*count].contiguous(),
                                          p2g=True, g2g=False)
                g_g_score = self.embed[j](gallery_x[:,i*count:(i+1)*count].contiguous(),
                                          gallery_x[:,i*count:(i+1)*count].contiguous(),
                                          p2g=False, g2g=True)

                outputs.append(random_walk_compute(p_g_score, g_g_score, self.alpha))

        outputs = torch.cat(outputs, 0)
        return outputs
