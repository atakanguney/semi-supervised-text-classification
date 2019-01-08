import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .prodlda import ProdLDA
from .prodlda import Encoder, Decoder


class Classifier(nn.Module):
    def __init__(self, dims):
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x


class ProdLDADeepGenerativeModel(ProdLDA):
    def __init__(self, dims, prior_mean, prior_var):
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(ProdLDADeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim], prior_mean, prior_var)

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, [], x_dim])

        self.classifier = Classifier([x_dim, h_dim[0], self.y_dim])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))

        self.kl_divergence = self._kld(z_mu, z_log_var)

        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu


    def classify(self, x):
        logits = self.classifier(x)
        return logits


    def sample(self, z, y):
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))

        return x
