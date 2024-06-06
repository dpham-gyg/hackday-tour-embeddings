from __future__ import print_function
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, n_tours: int):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n_tours, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, n_tours)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(y, x, mu, logvar):
    BCE = F.binary_cross_entropy(y, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model: VAE, epochs: int, train_dl: DataLoader) -> nn.Module:
    for epoch in range(epochs):
        current_loss = train_one_epoch(model, train_dl)
        print(f"Epoch: {epoch}, Loss: {current_loss}")

    return model


def train_one_epoch(model, train_loader):
    device = torch.device("cuda")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    train_loss = 0

    for _, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss
