import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import os

# P = np.load('/projects/perinatal/peridata/cdmri2018/SHARDbasis-prisma-st.npz')
# Bfull_p = P['arr_0']
# Bsh_p = P['arr_1']
# Bshard_p = P['arr_2']
# effect_p = P['arr_3']

def get_sh_slice(r,l):
    roffset = [0,1,29]
    i = 0
    i += roffset[r]

    for _l in [0,2,4,6,8,10,12]:
        if l == _l:
            break
        i += 2*_l+1
    return (i, i+2*l+1)

def shard2sh(shard, Bshard):
    ''' batch shard to spherical harmonics'''
    b, c = shard.shape[:2]
    return torch.bmm(Bshard.unsqueeze(0).expand(b,*Bshard.shape), shard.view(b,c,-1)).view(*shard.shape) # .contiguous()

def sh2rish(sh,r,l):
    ''' batch spherical harmonics to RISH '''
    sl = get_sh_slice(r,l)
    b = sh.shape[0]
    return sh[-1,sl[0]:sl[1],:,:,:].pow(2).sum(0).view(b,1,*sh.shape[2:])

def shard2sh_np(shard, Bshard):
    return np.dot(shard, Bshard.T)

def sh2rish_np(arr, shell, l):
    sl = get_sh_slice(shell,l)
    return np.power(arr[...,sl[0]:sl[1]],2).sum(3)

def geo_mean(a):
    a = torch.log(a)
    return torch.exp(a.sum()/len(a))


class RISH2Metric(nn.Module):
    def __init__(self, num_layers, dim=6, dim_inter=6, output_dim=1, activation='selu', norm='bn'):
        super(RISH2Metric, self).__init__()

        self.dim = dim
        self.output_dim = output_dim

        assert num_layers > 1, "need intermediate layers: {}".format(num_layers)
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'preluCh':
            self.activation = nn.PReLU(dim)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True) # Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize normalization
        norm_dim = dim
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)


        model = []
#         model += [nn.Linear(dim, dim_inter)]
        model += [nn.Conv3d(dim, dim_inter, kernel_size=1, stride=1, padding=0,dilation=1, bias=True)]
        if self.activation is not None:
            model += [self.activation]
        for i in range(num_layers - 1):
            model += [nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0,dilation=1, bias=True)]
            if self.activation is not None:
                model += [self.activation]
        model += [nn.Conv3d(dim, output_dim, kernel_size=1, stride=1, padding=0,dilation=1, bias=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        shape = x.shape
        if self.norm is not None:
            x = self.norm(x)
            print (x.shape)
        print (x.shape)
        x = self.model(x)
        print (x.shape)
        return x

class RISH2MetricFC(nn.Module):
    def __init__(self, num_layers, dim=9, dim_inter=32, output_dim=1, activation='relu', norm='bn', dropout=True, final_activation='none', using_batch_statistics=True):
        super(RISH2MetricFC, self).__init__()

        self.dim = dim
        self.output_dim = output_dim

        assert num_layers > 1, "need intermediate layers: {}".format(num_layers)
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'preluCh':
            self.activation = nn.PReLU(dim)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True) # Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize normalization
        norm_dim = dim
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(norm_dim, track_running_stats = not using_batch_statistics)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(norm_dim, track_running_stats = not using_batch_statistics)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)


        model = []
        model += [nn.Linear(dim, dim_inter)]
#         model += [nn.Conv3d(dim, dim_inter, kernel_size=1, stride=1, padding=0,dilation=1, bias=True)]
        if self.activation is not None:
            model += [self.activation]
        for i in range(num_layers - 1):
#             model += [nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0,dilation=1, bias=True)]
            model += [nn.Linear(dim_inter, dim_inter)]
            if self.activation is not None:
                model += [self.activation]
            if dropout and i % 2 == 0:
                model += [nn.Dropout(p=0.3, inplace=False)] # AlphaDropout
#         model += [nn.Conv3d(dim, output_dim, kernel_size=1, stride=1, padding=0,dilation=1, bias=True)]
        model += [nn.Linear(dim_inter, output_dim)]
        if final_activation is 'sigmoid':
            model += [nn.modules.activation.Sigmoid()]
        elif final_activation is not 'none':
            assert 0, "Unsupported final_activation: {}".format(final_activation)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        shape = x.shape
        if self.norm is not None:
            x = self.norm(x)
        shape = x.shape
        x = x.contiguous().view(shape[0],shape[1],-1).transpose(1,2)
        x = self.model(x)
        return x.transpose(2,1).view(shape[0],self.output_dim,*shape[2:])

class MovingAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.c = 0
        self.n = 0

    def update(self, f):
        self.c += f
        self.n += 1

    def __iadd__(self, f):
        self.update(f)
        return self

    def __call__(self):
        if self.n == 0:
            return 0
        return self.c / self.n


class WeightedShardMSELoss(object):
    def __init__(self, cuda=False, power=1.0):
        self.mse = nn.MSELoss()
        self.cuda = cuda
        self.power = power

        P = np.load(os.path.expanduser('/home/mp14/cdmri2018_data/SHARDbasis-prisma-st.npz'))
        self.effect_p = Variable(torch.from_numpy(P['arr_3']).float(), requires_grad=False)
        self.effect_p /= geo_mean(self.effect_p)

        C = np.load(os.path.expanduser('/home/mp14/cdmri2018_data/SHARDbasis-connectom-st.npz'))
        self.effect_c = Variable(torch.from_numpy(C['arr_3']).float(), requires_grad=False)
        self.effect_c /= geo_mean(self.effect_c)

        if self.power != 1.0:
            self.effect_c = torch.pow(self.effect_c, self.power)

        if self.cuda:
            self.effect_p = self.effect_p.cuda()
            self.effect_c = self.effect_c.cuda()

    def __call__(self, S, T, M=None):

        Ec = self.effect_c.view(1,self.effect_c.numel(),1,1,1).expand(S.shape[0],self.effect_c.numel(),*S.shape[-3:])

        assert Ec.is_cuda == S.is_cuda == T.is_cuda, str([Ec.is_cuda, S.is_cuda, T.is_cuda])

        if M is None:
            return self.mse(torch.mul(S, Ec), torch.mul(T, Ec))
        return self.mse(torch.mul(S, Ec)[M.expand_as(S)>0.5], torch.mul(T, Ec)[M.expand_as(S)>0.5])


class MSEandFALoss(object):
    def __init__(self, weights=[1.0, 1.0], SHARD_effect_weighted=False, cuda=False):
        self.mse = nn.MSELoss()
        self.r2m = RISH2MetricFC(6, dropout=True, activation='relu', final_activation='none', using_batch_statistics=True)
        self.cuda = cuda

        self.weights = weights
        self.SHARD_effect_weighted = SHARD_effect_weighted


        if cuda:
            self.r2m.load_state_dict(torch.load('/home/mp14/cdmri2018_data/metric_models/metric_dt_fa.pth.tar0.028178'))
        else:
            self.r2m.load_state_dict(torch.load('/home/mp14/cdmri2018_data/metric_models/metric_dt_fa.pth.tar0.028178', map_location=lambda storage, loc: storage))
        self.r2m.eval()
        for c in self.r2m.children():
            for param in c.parameters():
                param.requires_grad = False

        P = np.load(os.path.expanduser('/home/mp14/cdmri2018_data/SHARDbasis-prisma-st.npz'))
        self.Bshard_p = torch.from_numpy(P['arr_2']).float()
        self.effect_p = Variable(torch.from_numpy(P['arr_3']).float(), requires_grad=False)
        self.effect_p /= self.effect_p.max()

        C = np.load(os.path.expanduser('/home/mp14/cdmri2018_data/SHARDbasis-connectom-st.npz'))
        self.Bshard_c = torch.from_numpy(C['arr_2']).float()
        self.effect_c = Variable(torch.from_numpy(C['arr_3']).float(), requires_grad=False)
        self.effect_c /= self.effect_c.max()

        if self.cuda:
            self.Bshard_p = self.Bshard_p.cuda()
            self.Bshard_c = self.Bshard_c.cuda()
            self.r2m = self.r2m.cuda()
            self.effect_p = self.effect_p.cuda()
            self.effect_c = self.effect_c.cuda()

    def __shard2fa(self, shard, M=None, prisma=True):
        if prisma:
            SH = shard2sh(shard, self.Bshard_p)
        else:
            SH = shard2sh(shard, self.Bshard_c)
        RISH = torch.cat([sh2rish(SH, r, l) for r, l in [(0,0),(1,0),(2,0),(1,2),(1,4),(1,6),(2,2),(2,4),(2,6)]], dim=1)
        if self.cuda :
            RISH = RISH.cuda()
        FA = self.r2m(RISH)
        if M is not None:
            FA.masked_fill_(MASK<0.5,0)
#         nnan = FA != FA
#         if nnan.any() > 0:
#             FA[nnan] = 0
        return FA

    def __call__(self, S, T, M=None):
        Sfa = self.__shard2fa(S, M=M, prisma=True)
        Tfa = self.__shard2fa(T, M=M, prisma=False)
        if self.SHARD_effect_weighted:
            Ec = self.effect_c.view(1,self.effect_c.numel(),1,1,1).expand(S.shape[0],-1,-1,*S.shape[-3:])
            if M is None:
                return self.weights[0] * self.mse(torch.mul(S, Ec), torch.mul(T, Ec)) + self.weights[1] * self.mse(Sfa, Tfa)
            assert 0, 'Masked weighted cost not implemented'

        if M is None:
            return self.weights[0] * self.mse(S,T) + self.weights[1] * self.mse(Sfa, Tfa)
        return self.weights[0] * self.mse(S[M>0.5],T[M>0.5]) + self.weights[1] * self.mse(Sfa[M>0.5],Tfa[M>0.5])

# r2m = RISH2Metric(8)

# target_sh = shard2sh(target, Bshard)
# prediction_sh = shard2sh(prediction, Bshard)

# rlsw = [ (1,2,1.0), (1,4,1.0), (2,2,1.0), (2,2,1.0)]

# loss_fn = nn.SmoothL1Loss()
# losses = [ weight * loss_fn(prediction, sh2rish(target_sh,r,l)) for r, l, weight in rlsw ]
# loss = sum(losses)

# sh_p_t = shard2sh(shard_t, Bshard_t)


# 1.2k:
# a = np.array([11800., 2050, 160, 45])
# 3k:
# b = [3000., 1300, 400, 93]

