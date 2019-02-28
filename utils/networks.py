import torch.nn as nn
import torch.nn.functional as F
import torch as th


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=None, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if hidden_dim is None:
            hidden_dim = [128]
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1.)
            self.in_fn.bias.data.fill_(0.)
        else:
            self.in_fn = lambda x: x
        self.fc = nn.ModuleList()
        last_dim = input_dim
        for dim_i in hidden_dim:
            self.fc.append(nn.Linear(last_dim, dim_i))
            last_dim = dim_i
        self.fc.append(nn.Linear(last_dim, out_dim))
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc[-1].weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X, return_preactivations=False):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h = self.nonlin(self.fc[0](self.in_fn(X)))

        for i in range(1, len(self.fc) - 1):
            h = self.nonlin(self.fc[i](h))
        preactivations = self.fc[-1](h)
        out = self.out_fn(preactivations)
        if return_preactivations:
            return out, preactivations
        else:
            return out


class SinMLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=None, nonlin=F.relu, norm_in=False):
        """
        Inputs:`
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(SinMLPNetwork, self).__init__()

        if hidden_dim is None:
            hidden_dim = [128]
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1.)
            self.in_fn.bias.data.fill_(0.)
        else:
            self.in_fn = lambda x: x
        self.fc = nn.ModuleList()
        last_dim = input_dim
        for dim_i in hidden_dim:
            self.fc.append(nn.Linear(last_dim, dim_i))
            last_dim = dim_i
        self.fc.append(nn.Linear(last_dim, out_dim))
        self.fc.append(nn.Linear(last_dim, out_dim))
        self.nonlin = nonlin
        self.out_fn = lambda x: x

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, X, return_preactivations=False):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (mu, std)
        """
        h = self.nonlin(self.fc[0](self.in_fn(X)))

        for i in range(1, len(self.fc) - 2):
            h = self.nonlin(self.fc[i](h))
        mu = self.fc[-2](h)
        logvar = self.fc[-1](h)
        z = self.reparameterize(mu, logvar)
        return z
