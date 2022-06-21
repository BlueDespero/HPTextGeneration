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

        n_vocab = dataset.uniq_words
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


def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def word_generating_procedure():
    # Here is how to use this function for top-p sampling
    temperature = 1.0
    top_p = 0.9

    # Get logits with a forward pass in our model (input is pre-defined)
    logits = model(input)

    # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
    logits = logits[0, -1, :] / temperature
    filtered_logits = top_p_filtering(logits, top_p=top_p)

    # Sample from the filtered distribution
    probabilities = nn.functional.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1)


batch_size = 512
max_epochs = 10

dataset = Dataset()
model = Model(dataset)
model.to(device)

train(dataset, model, batch_size, max_epochs)
torch.save(model.state_dict(), os.path.join(ROOT_DIR, "data_processed", "model"))
print(predict(dataset, model, text='Wand was risen'))
