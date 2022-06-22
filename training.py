import os.path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from word_tokenization import Dataset
from definitions import ROOT_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = dataset.n_tokens
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


def train(dataset: Dataset, model, batch_size, max_epochs):
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(dataset.seq_size)

        for batch, i in enumerate(range(0, len(dataset), batch_size)):
            x, y = dataset.get_batch(i, batch_size)
            if isinstance(x, int):
                continue
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
            if batch % 100 == 0:
                torch.save(model.state_dict(), os.path.join(ROOT_DIR, "data_processed", "model"))



if __name__ =="__main__":
    batch_size = 512
    max_epochs = 10

    dataset = Dataset()
    model = Model(dataset)
    model.to(device)

    train(dataset, model, batch_size, max_epochs)
    torch.save(model.state_dict(), os.path.join(ROOT_DIR, "data_processed", "model"))
