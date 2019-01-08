import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from layers import GaussianSample
from inference import log_gaussian, log_standard_gaussian


class Encoder(nn.Module):
    def __init__(self, dims, sample_layer=GaussianSample):
        super(Encoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
    
        self.hidden = nn.ModuleList(linear_layers)

        self.sample = sample_layer(h_dim[-1], z_dim)

        self.mean_norm = nn.BatchNorm1d(z_dim)
        self.logvar_norm = nn.BatchNorm1d(z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))

        z, mean, logvar = self.sample(x)
        mean = self.mean_norm(mean)
        logvar = self.logvar_norm(logvar)

        return z, mean, logvar


class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims

        if h_dim:
            neurons = [z_dim, *h_dim, x_dim]
        else:
            neurons = [z_dim, x_dim]

        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]

        self.activation = nn.Softmax()
        self.drop = nn.Dropout(0.2)

        self.hidden = nn.ModuleList(linear_layers)

        self.norm = nn.BatchNorm1d(x_dim)
        self.output_activation = nn.Softmax()

    def forward(self, x):
        x = self.activation(x)
        x = self.drop(x)
    
        for layer in self.hidden:
            x = F.relu(layer(x))
    
        x = self.norm(x)

        return self.output_activation(x)


class ProdLDA(nn.Module):
    def __init__(self, dims, prior_mean, prior_var):
        super(ProdLDA, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.flow = None

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, [], x_dim])

        prior_mean = torch.Tensor(1, self.z_dim).fill_(prior_mean)
        prior_var = torch.Tensor(1, self.z_dim).fill_(prior_var)
        prior_logvar = prior_var.log()
        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_var", prior_var)
        self.register_buffer("prior_logvar", prior_logvar)

        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def _kld(self, mu, log_var):
        prior_mean = Variable(self.prior_mean).expand_as(mu)
        prior_var = Variable(self.prior_var).expand_as(mu)
        prior_logvar = Variable(self.prior_logvar).expand_as(mu)

        var_division = log_var.exp() / prior_var
        diff = mu - prior_mean
        diff_term = diff * diff / prior_var

        logvar_division = prior_logvar - log_var

        return 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.z_dim)

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        z, z_mu, z_log_var = self.encoder(x)

        self.kl_divergence =self._kld(z_mu, z_log_var)

        x_mu = self.decoder(z)

        return x_mu

    def sample(self, z):
        return self.decoder(z)

