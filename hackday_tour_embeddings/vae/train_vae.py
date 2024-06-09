from __future__ import print_function

from functools import partial
from typing import Dict

import torch
from ignite.engine import (
    Engine,
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import ModelCheckpoint, TensorboardLogger, global_step_from_engine
from ignite.metrics import ROC_AUC
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hackday_tour_embeddings.data_preparation import data_loading

TOUR_EMB_SIZE = 300
# N_TOURS = 135286
N_TOURS = 106114
# N_TOURS = 123893


MODEL_ARTFACT_PATH = "/mnt/data/duy.pham/hackdays-24-06/models/"


class VAE(nn.Module):
    def __init__(self, n_tours: int, bert_emb_matrix: torch.Tensor):
        super(VAE, self).__init__()

        assert bert_emb_matrix.shape == torch.Size(
            [
                N_TOURS,
                data_loading.BERT_EMB_SIZE,
            ]
        ), f" bert shape is {bert_emb_matrix.shape}"

        self.tour_emb_fc1 = nn.Linear(data_loading.BERT_EMB_SIZE, TOUR_EMB_SIZE)
        self.tour_emb_fc2 = nn.Linear(TOUR_EMB_SIZE, 1)

        self.fc1 = nn.Linear(n_tours, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, n_tours)
        self.n_tours = n_tours

        # Learnable tour embeddings
        # self.tour_emb_matrix = torch.nn.Embedding(n_tours, TOUR_EMB_SIZE)

        # Freeze bert pretrained embeddings
        self.bert_emb_matrix = torch.nn.Embedding.from_pretrained(
            bert_emb_matrix
        ).requires_grad_(False)

    def embedding_mix(self, bert_embs):
        """
        tour_embs: (batch_size, n_tours, TOUR_EMB_SIZE)
        bert_embs: (batch_size, n_tours, BERT_EMB_SIZE)
        """
        # concat: (batch_size, n_tours, TOUR_EMB_SIZE + BERT_EMB_SIZE)
        # concat = torch.cat([tour_embs, bert_embs], dim=-1)
        # output (batch_size, n_tours, 1)
        tours_embs = self.tour_emb_fc1(bert_embs)
        return self.tour_emb_fc2(tours_embs).squeeze()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # tour_embs = self.tour_emb_matrix(x)
        # bert_embs = self.bert_emb_matrix(x.to(torch.int32))
        # x = self.embedding_mix(bert_embs)
        # mu, logvar = self.encode(x.view(-1, self.n_tours))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(y, x, mu, logvar):
    # print(f"computing loss between {y} and {x}")
    BCE = F.binary_cross_entropy(y, x, reduction="sum")
    # print("BCE", BCE.item())
    # reconstruction_loss = F.mse_loss(y, x.view(-1, N_TOURS), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print("KLD", KLD.item())

    return BCE + KLD
    # return reconstruction_loss + KLD


def train(epochs: int, train_dl: DataLoader, n_tours, bert_emb_matrix) -> nn.Module:
    device = torch.device("cuda")
    model = VAE(n_tours=n_tours, bert_emb_matrix=bert_emb_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        print(f"Start training epoch: {epoch}")

        train_loss = 0
        for batch_idx, batch in enumerate(train_dl):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"batch {batch_idx} loss = {train_loss / (batch_idx + 1)}")

        print(f"Epoch: {epoch}, Loss: {train_loss / (batch_idx + 1)}")

    torch.save(model.state_dict(), MODEL_ARTFACT_PATH + "/vae-small/")
    print("Model saved to ", MODEL_ARTFACT_PATH + "/vae-small/")
    return model


def train_with_ignite(
    epochs: int,
    train_dl: DataLoader,
    validation_dl: DataLoader,
    n_tours: int,
    bert_emb_matrix: torch.Tensor,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(n_tours=n_tours, bert_emb_matrix=bert_emb_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = Engine(
        partial(train_step, model=model, optimizer=optimizer, device=device)
    )

    evaluator = create_supervised_evaluator(model, metrics={"auc": ROC_AUC()})

    model_checkpoint = ModelCheckpoint(
        "checkpoint",
        n_saved=2,
        filename_prefix="best",
        score_function=lambda engine: engine.state.metrics["auc"],
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer),
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        partial(validation, evaluator=evaluator, validation_dl=validation_dl),
    )

    evaluator.add_event_handler(
        Events.COMPLETED,
        model_checkpoint,
        {"model": model},
    )

    tb_logger = TensorboardLogger(log_dir="tensorboard")

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="evaluation",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    trainer.run(train_dl, max_epochs=epochs)

    tb_logger.close()


def train_step(engine, batch, optimizer, model, device):
    batch = batch.to(device)
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(batch)
    loss = loss_function(recon_batch, batch, mu, logvar)
    # print(f"loss = {loss.item()}")
    loss.backward()
    optimizer.step()
    return loss.item()


def validation(trainer, evaluator, validation_dl):
    state = evaluator.run(validation_dl)
    print(trainer.state.epoch, state.metrics)
