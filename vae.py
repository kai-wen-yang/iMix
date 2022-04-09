from __future__ import print_function
import abc
import os
import math
import pdb
import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Function, Variable
import functools


class MLP(nn.Module):
    def __init__(
        self,
        in_features=2048,
        hidden_layers=[],
        num_classes=10,
        activation="relu",
        bn=True,
        dropout=0.0,
    ):
        super().__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        assert len(hidden_layers) > 0
        self.out_features = hidden_layers[-1]

        mlp = []

        if activation == "relu":
            act_fn = functools.partial(nn.ReLU, inplace=True)
        elif activation == "leaky_relu":
            act_fn = functools.partial(nn.LeakyReLU, inplace=True)
        else:
            raise NotImplementedError

        for hidden_dim in hidden_layers:
            mlp += [nn.Linear(in_features, hidden_dim)]
            if bn:
                mlp += [nn.BatchNorm1d(hidden_dim)]
            mlp += [act_fn()]
            if dropout > 0:
                mlp += [nn.Dropout(dropout)]
            in_features = hidden_dim
        mlp += [nn.Linear(self.out_features, num_classes)]
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class CVAE_cifar_latent(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_cifar_latent, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )

        self.encoder_y = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.encoder_n = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(3)
        self.f = 8
        self.d = d
        self.z = z
        self.fc_y1 = nn.Linear(d * self.f ** 2, self.z)
        self.fc_y2 = nn.Linear(d * self.f ** 2, self.z)
        self.fc_n1 = nn.Linear(d * self.f ** 2, self.z)
        self.fc_n2 = nn.Linear(d * self.f ** 2, self.z)
        self.fc = nn.Linear(2 * self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        hy = self.encoder_y(h)
        hn = self.encoder_n(h)
        hy = hy.view(-1, self.d * self.f ** 2)
        hn = hn.view(-1, self.d * self.f ** 2)
        return self.fc_y1(hy), self.fc_y2(hy), self.fc_n1(hn), self.fc_n2(hn)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x, decode=False):
        if decode:
            z_projected = self.fc(x)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            return gx
        else:
            mu_y, logvar_y, mu_n, logvar_n = self.encode(x)
            z_y = self.reparameterize(mu_y, logvar_y)
            z_n = self.reparameterize(mu_n, logvar_n)
            z = torch.cat((z_y, z_n), dim=1)
            z_projected = self.fc(z)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
        return z_y, z_n, gx,  mu_y, logvar_y, mu_n, logvar_n



class CVAE_ImagenetDown_withbn(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_ImagenetDown_withbn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(3)
        self.f = 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x, decode=False):
        if decode:
            z_projected = self.fc21(x)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            return gx
        else:
            _, mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_projected = self.fc21(z)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
        return z, gx, mu, logvar


class CVAE_miniImagenet_withbn(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_miniImagenet_withbn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(3)
        self.f = 10
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        # if self.training:
        #     std = logvar.mul(0.5).exp_()
        #     eps = std.new(std.size()).normal_()
        #     return eps.mul(std).add_(mu)
        # else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x, decode=False):
        if decode:
            z_projected = self.fc21(x)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            return gx
        else:
            _, mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_projected = self.fc21(z)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
        return z, gx, mu, logvar


class CVAE_tinyimagenet_withbn(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_tinyimagenet_withbn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(3)
        self.f = 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x, decode=False):
        if decode:
            z_projected = self.fc21(x)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            return gx
        else:
            _, mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_projected = self.fc21(z)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
        return z, gx, mu, logvar


class CVAE_cifar_withbn(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_cifar_withbn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(3)
        self.f = 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x, decode=False):
        if decode:
            z_projected = self.fc21(x)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            return gx
        else:
            _, mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_projected = self.fc21(z)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
        return z, gx, mu, logvar


class CVAE_imagenet_withbn(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_imagenet_withbn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 16, d // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 8, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d)
        )

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 4, d // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 8),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 8, d // 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 16, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.bn = nn.BatchNorm2d(3)
        self.f = 7
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x, decode=False):
        if decode:
            z_projected = self.fc21(x)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            return gx
        else:
            _, mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_projected = self.fc21(z)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
        return z, gx, mu, logvar


class VQVAE_imagenet(nn.Module):
    def __init__(self, d, k=10, num_channels=3, **kwargs):
        super(VQVAE_imagenet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d//4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(d//4, d//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(inplace=True),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d//2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d//2, d // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d//4, num_channels, kernel_size=4, stride=2, padding=1),
        )
        self.d = d
        self.emb = NearestEmbed(k, d)

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)
        self.L_bn = nn.BatchNorm2d(num_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x, decode=False):
        if decode:
            l = self.decode(x)
            gx = self.L_bn(l)

            return gx
        else:
            z_e = self.encode(x)

            z_q, _ = self.emb(z_e, weight_sg=True)
            emb, _ = self.emb(z_e.detach())

            l = self.decode(z_q)
            gx = self.L_bn(l)
            return z_q, gx, z_e, emb


class CVAE_imagenet_withbn_v2(AbstractAutoEncoder):
    def __init__(self, d, z,  **kwargs):
        super(CVAE_imagenet_withbn_v2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=4, padding=0, bias=False),
        )
        self.bn = nn.BatchNorm2d(3)
        self.f = 14
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x, decode=False):
        if decode:
            z_projected = self.fc21(x)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
            return gx
        else:
            _, mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_projected = self.fc21(z)
            gx = self.decode(z_projected)
            gx = self.bn(gx)
        return z, gx, mu, logvar


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *list(input.shape[2:]) ,input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)