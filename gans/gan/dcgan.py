import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transfroms
import torch.nn.functional as F

torch.manual_seed(42)


class Generator(nn.Module):
    def __init__(self, nc=3, dim_z=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, nc, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class WrapD(nn.Module):
    def __init__(self, D):
        super(WrapD, self).__init__()
        self.D = D

    def forward(self, x):
        prob = self.D(x)
        return prob

    def ar(self, x_s, x_p):
        d_s = self.D(x_s).cpu()
        d_p = self.D(x_p).cpu()
        log_prob = torch.log(1. - d_s) + torch.log(d_p) - torch.log(d_s) - torch.log(1 - d_p)
        log_prob = torch.clamp(log_prob, max=0.0).view(log_prob.size(0), )
        return torch.exp(log_prob)

class WrapCriticD(nn.Module):
    def __init__(self, D):
        super(WrapCriticD, self).__init__()
        self.D = D
        self.nonlinear = nn.Sigmoid()

    def forward(self, x):
        logit = self.D(x)
        prob = self.nonlinear(logit)
        return prob

    def ar(self, x_s, x_p):
        d_s = self.forward(x_s).cpu()
        d_p = self.forward(x_p).cpu()
        log_prob = torch.log(1. - d_s) + torch.log(d_p) - torch.log(d_s) - torch.log(1 - d_p)
        log_prob = torch.clamp(log_prob, max=0.0).view(log_prob.size(0), )
        return torch.exp(log_prob)
