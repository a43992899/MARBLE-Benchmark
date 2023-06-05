import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class GE2ELoss(nn.Module):
    w: nn.Parameter
    b: nn.Parameter
    criterion: nn.CrossEntropyLoss

    def __init__(self, init_w=10.0, init_b=-5.0):
        super(GE2ELoss, self).__init__()

        self.test_normalize = True

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: Tensor, label=None):
        assert x.size()[1] >= 2

        gsize = x.size()[1]
        centroids = torch.mean(x, 1)
        stepsize = x.size()[0]

        cos_sim_matrix = []

        for ii in range(0, gsize):
            idx = [*range(0, gsize)]
            idx.remove(ii)
            exc_centroids = torch.mean(x[:, idx, :], 1)
            cos_sim_diag = F.cosine_similarity(x[:, ii, :], exc_centroids)
            cos_sim = F.cosine_similarity(x[:, ii, :].unsqueeze(-1), centroids.unsqueeze(-1).transpose(0, 2))
            cos_sim[range(0, stepsize), range(0, stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim, 1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix, dim=1)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(np.asarray(range(0, stepsize))).cuda()
        nloss = self.criterion(cos_sim_matrix.view(-1, stepsize),
                               torch.repeat_interleave(label, repeats=gsize, dim=0).cuda())
        prec1 = accuracy(cos_sim_matrix.view(-1, stepsize).detach(),
                         torch.repeat_interleave(label, repeats=gsize, dim=0).detach(), topk=(1,))[0]
        return nloss, prec1


class AAMSoftmax(nn.Module):
    test_normalize: bool
    m: float
    s: float
    in_feats: int
    weight: nn.Parameter
    ce: nn.CrossEntropyLoss
    easy_margin: bool

    cos_m: float
    sin_m: float
    th: float
    mm: float

    def __init__(self, input_features: int, num_classes: int, margin=0.3, scale=15, easy_margin=False):
        super(AAMSoftmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = input_features
        self.ce = nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, input_features), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def directly_predict(self, x: Tensor):
        cosine_theta = F.linear(F.normalize(x), F.normalize(self.weight))  # \cos_\theta
        return cosine_theta

    def predict(self, x: Tensor, label: Tensor):
        assert x.size(0) == label.size(0)
        assert x.size(1) == self.in_feats
        cosine_theta = self.directly_predict(x)
        sine_theta = torch.sqrt((1.0 - torch.mul(cosine_theta, cosine_theta)).clamp(0, 1))  # \sin_\theta
        cos_theta_plus_m = cosine_theta * self.cos_m - sine_theta * self.sin_m  # \cos(\theta + m)

        if self.easy_margin:
            phi = torch.where(cosine_theta > 0, cos_theta_plus_m, cosine_theta)
        else:
            phi = torch.where((cosine_theta - self.th) > 0, cos_theta_plus_m, cosine_theta - self.mm)

        one_hot = torch.zeros_like(cosine_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output: Tensor = one_hot * phi + (1.0 - one_hot) * cosine_theta
        output = output * self.s

        return output

    def forward(self, x: Tensor, label: Tensor):
        output = self.predict(x, label)
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


class SubcenterArcMarginProduct(nn.Module):
    """Modified from:
    https://github.com/ronghuaiyang/arcface-pytorch/blob/
    47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
    """

    s: float
    m: float
    K: int

    easy_margin: bool

    def __init__(self, in_features, out_features, K=3, s=30.0, m=0.50, easy_margin=False):
        super(SubcenterArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(out_features * self.K, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.ce_loss = nn.CrossEntropyLoss()

    def directly_predict(self, x: Tensor):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            cosine, _ = torch.max(cosine, axis=2)

        return cosine

    def predict(self, x: Tensor, label: Tensor):
        cosine = self.directly_predict(x)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def forward(self, x: Tensor, label: Tensor):
        output = self.predict(x, label)
        loss = self.ce_loss(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
