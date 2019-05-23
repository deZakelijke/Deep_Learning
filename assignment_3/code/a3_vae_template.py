import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.nn.functional import binary_cross_entropy
from datasets.bmnist import bmnist


MNIST_SIZE = 28 * 28

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.fc_1   = nn.Linear(MNIST_SIZE, hidden_dim)
        self.fc_mu  = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        self.tanh   = nn.Tanh()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        h1 = self.tanh(self.fc_1(input))
        mean = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.fc_1 = nn.Linear(z_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, MNIST_SIZE)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

    def forward(self, input):
        """
        Perform forward pass of decoder

        Returns mean with shape [batch_size, 784].
        """
        h1 = self.tanh(self.fc_1(input))
        mean = self.sigm(self.fc_2(h1))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mu, logvar = self.encoder(input)
        z = self.reparameterize(mu, logvar)
        recon_input = self.decoder(z)

        average_negative_elbo = self.calc_average_neg_elbo(recon_input, input, mu, logvar)
        return average_negative_elbo

    def calc_average_neg_elbo(self, recon_input, input, mu, logvar):
        reconstruction_loss = binary_cross_entropy(recon_input, input, reduction='mean')
        regularizing_loss = -0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)
        return torch.sum(reconstruction_loss - regularizing_loss).div(input.shape[0])


    def reparameterize(self, mu, logvar):
        std = logvar.exp_()
        eps = torch.FloatTensor(std.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        z = torch.randn((n_samples, self.z_dim))
        if torch.cuda.is_available:
            z = z.cuda()
        im_means = self.decoder(z)
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims.cpu(), im_means.cpu()


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0
    for data_sample in data:
        model.zero_grad()
        elbo = model(data_sample.view(-1, MNIST_SIZE))

        if model.training:
            elbo.backward()
            optimizer.step()

        average_epoch_elbo += elbo

    return average_epoch_elbo / len(data)


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main(ARGS):
    learning_rate = 1e-4
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')
    torch.save(model, f"VAE-model_{ARGS.z_dim}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main(ARGS)
