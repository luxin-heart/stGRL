import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


###################################ZINB###################################
###################################ZINB###################################
###################################ZINB###################################

class ZINBLoss(nn.Module):

    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0, device=None):
        eps = 1e-10
        scale_factor = torch.Tensor([1.0]).to(device)
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


###################################ZINB###################################
###################################ZINB###################################
###################################ZINB###################################


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)


class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        # self.weight3 = Parameter(torch.FloatTensor(self.out_features, self.out_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

        ##############################################################
        ##############################################################
        self._dec_mean = nn.Sequential(nn.Linear(self.in_features, self.in_features), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(self.in_features, self.in_features), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(self.in_features, self.in_features), nn.Sigmoid())
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(self.out_features),
            nn.Dropout(0.3),
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features)
        )

        self.zinb_loss = ZINBLoss().cuda()

        ##############################################################
        ##############################################################

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)

        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)

        #############################################
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)
        zinb_loss = self.zinb_loss
        #############################################

        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)



        dec = self.decoder(emb)
        dec_a = self.decoder(emb_a)


        return hiden_emb, h, zinb_loss, _mean, _disp, _pi, dec, dec_a

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def contraction_loss(self, z, z_a):
        l1 = self.semi_loss(z, z_a)
        l2 = self.semi_loss(z_a, z)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret


class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """

    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder_sparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(self.out_features),
            nn.Dropout(0.3),
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features)
        )

        ##############################################################
        ##############################################################
        self._dec_mean = nn.Sequential(nn.Linear(self.in_features, self.in_features), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(self.in_features, self.in_features), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(self.in_features, self.in_features), nn.Sigmoid())

        self.zinb_loss = ZINBLoss().cuda()

        ##############################################################
        ##############################################################

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)

        hiden_emb = z
        emb = self.act(z)

        h = torch.mm(emb, self.weight2)
        h = torch.spmm(adj, h)

        #############################################
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)
        zinb_loss = self.zinb_loss
        #############################################

        # emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
        dec = self.decoder(emb)
        dec_a = self.decoder(emb_a)

        # g = self.read(emb, self.graph_neigh)
        # g = self.sigm(g)
        #
        # g_a = self.read(emb_a, self.graph_neigh)
        # g_a = self.sigm(g_a)
        #
        # ret = self.disc(g, emb, emb_a)
        # ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, zinb_loss, _mean, _disp, _pi, dec, dec_a

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def contraction_loss(self, z, z_a):
        l1 = self.semi_loss(z, z_a)
        l2 = self.semi_loss(z_a, z)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret




