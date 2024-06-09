from __future__ import print_function
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F


BERT_EMB_SIZE = 768
TOUR_EMB_SIZE = 300
N_TOURS = 135286


class VAE(nn.Module):
    def __init__(self, n_tours: int):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n_tours, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, n_tours)
        self.n_tours = n_tours

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
        mu, logvar = self.encode(x.view(-1, self.n_tours))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(y, x, mu, logvar):
    BCE = F.binary_cross_entropy(y, x.view(-1, N_TOURS), reduction='sum')
    # reconstruction_loss = F.mse_loss(y, x.view(-1, N_TOURS), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
    # return reconstruction_loss + KLD


def train(epochs: int, train_dl: DataLoader, n_tours) -> nn.Module:
    device = torch.device("cpu")
    model = VAE(n_tours=n_tours).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        print(f"Start training epoch: {epoch}")

        train_loss = 0
        for batch_idx, data in enumerate(train_dl):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 1000 == 0:
                print(f"batch {batch_idx} loss = {train_loss / 1000}")

        print(f"Epoch: {epoch}, Loss: {train_loss}")

    return model
