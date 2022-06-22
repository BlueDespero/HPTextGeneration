import os.path
import string

import nltk
import numpy as np
import torch
from nltk import RegexpTokenizer
from torch import nn

from definitions import ROOT_DIR
from training import Model
from word_tokenization import Dataset


def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def verify_text(text, dataset):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    bigrams = [b for b in zip(words[:-1], words[
                                          1:])]  # All the bigrams in the generated sequence exist in the books. This also assures the words are correct.
    for bigram in bigrams:
        if bigram not in dataset.bigrams:
            return False

    trigrams = [t for t in zip(words[:-2], words[1:-1], words[2:])]  # Trigrams are unique
    used_trigrams = set()
    for tri in trigrams:
        if tri in used_trigrams:
            return False
        else:
            used_trigrams.add(tri)

    return True


def predict(dataset, model, text, next_words=100, init_patience=5, unbounded=False):
    model.eval()
    target_length = len(text.split(" ")) + next_words
    init_text = text
    patience = init_patience

    tokens = dataset.tokenizer.encode(text).ids
    attention_size = len(tokens)
    state_h, state_c = model.init_state(attention_size)

    checkpoint = (text, state_h, state_c)

    while len(text.split(" ")) < target_length:

        if not unbounded:
            if not verify_text(text, dataset):
                patience -= 1
            elif text[-1] not in string.punctuation:
                patience = init_patience
                checkpoint = (text, state_h, state_c)

            if patience <= 0 and patience % 5 == 0:
                text, state_h, state_c = checkpoint
            if patience <= -1000:
                text = init_text
                state_h, state_c = model.init_state(attention_size)
                patience = init_patience

        tokens = dataset.tokenizer.encode(text).ids
        x = torch.tensor([tokens[-attention_size:]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[0][-1]
        last_word_logits = top_p_filtering(last_word_logits, top_p=0.5)  # Top p filtering is used
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        tokens.append(word_index)
        text = dataset.tokenizer.decode(tokens)

    return text


if __name__ == "__main__":
    dataset = Dataset()
    model = Model(dataset)
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "data_processed/model_finished")))
    print(predict(dataset, model, text='Harry had', unbounded=True))
    print(predict(dataset, model, text='Hagrid looked at his house'))
    print(predict(dataset, model, text='Snape did'))
    print(predict(dataset, model, text='Nicolas Flamel'))
