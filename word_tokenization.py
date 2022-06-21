import os
import pickle
import re
from typing import List

import nltk
import torch
from tokenizers import BertWordPieceTokenizer

from definitions import ROOT_DIR, VOCAB_SIZE, SEQ_SIZE

pattern = re.compile("^Page \\| .*")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def verify_line(line: str):
    if len(line) < 2:
        return False

    if pattern.match(line):
        return False

    return True


def load_corpus() -> List[str]:
    path_to_data = os.path.join(ROOT_DIR, "data")
    content = []

    for file in os.listdir(path_to_data):
        path_to_file = os.path.join(path_to_data, file)
        with open(path_to_file, 'r', encoding='utf-8') as book:
            content += [line for line in book.readlines() if verify_line(line)]

    return content


def check_number_of_words_in_corpus(corpus: List[str]):
    return sum([len(line.split()) for line in corpus])


def prepare_nltk():
    try:
        nltk.data.find('tokenizers/punkt.zip')
    except LookupError:
        nltk.download('punkt')


def get_tokenized_corpus(text, save=False):
    tokenized_corpus_path = os.path.join(ROOT_DIR, "data_processed", "tokenized_corpus.pickle")
    try:
        with open(tokenized_corpus_path, 'rb') as preprocessed_file:
            return pickle.load(preprocessed_file)
    except FileNotFoundError:
        tokenized_corpus = nltk.word_tokenize(text)
        if save:
            with open(tokenized_corpus_path, 'wb') as result_file:
                pickle.dump(tokenized_corpus, result_file)
        return tokenized_corpus


def get_corpus_of_sentences(text, save=False):
    sentences_corpus_path = os.path.join(ROOT_DIR, "data_processed", "sentences_in_corpus.pickle")
    try:
        with open(sentences_corpus_path, 'rb') as preprocessed_file:
            return pickle.load(preprocessed_file)
    except FileNotFoundError:
        sentences_in_corpus = nltk.sent_tokenize(text)
        if save:
            with open(sentences_corpus_path, 'wb') as result_file:
                pickle.dump(sentences_in_corpus, result_file)
        return sentences_in_corpus


def get_subword_tokenizer(tokenized_corpus, save=False):
    try:
        return BertWordPieceTokenizer(os.path.join(ROOT_DIR, "data_processed", "bert-vocab.txt"))
    except Exception:
        subword_tokenizer = BertWordPieceTokenizer(
            clean_text=False,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True,
        )

        files = [os.path.join(ROOT_DIR, "data", file) for file in os.listdir(os.path.join(ROOT_DIR, "data"))]

        subword_tokenizer.train(
            files,
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            show_progress=True,
            special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
            limit_alphabet=1000,
            wordpieces_prefix="##"
        )

        return subword_tokenizer


def get_tokenized_sentences(subword_tokenizer, sentences_in_corpus, save=False):
    tokenized_sentences_path = os.path.join(ROOT_DIR, "data_processed", "tokenized_sentences.pickle")
    try:
        with open(tokenized_sentences_path, 'rb') as preprocessed_file:
            return pickle.load(preprocessed_file)
    except FileNotFoundError:
        tokenized_sentences = subword_tokenizer.encode("".join(sentences_in_corpus))
        if save:
            with open(tokenized_sentences_path, 'wb') as result_file:
                pickle.dump(tokenized_sentences, result_file)
        return tokenized_sentences


def preprocessing_pipeline(save=False):
    prepare_nltk()
    corpus = load_corpus()
    full_text = " ".join(corpus)

    tokenized_corpus = get_tokenized_corpus(full_text, save)
    sentences_in_corpus = get_corpus_of_sentences(full_text, save)

    subword_tokenizer = get_subword_tokenizer(tokenized_corpus, save)
    tokenized_sentences = get_tokenized_sentences(subword_tokenizer, sentences_in_corpus)

    return tokenized_sentences, subword_tokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
    ):
        self.sentences, self.tokenizer = preprocessing_pipeline()
        self.uniq_words = VOCAB_SIZE
        self.seq_size = SEQ_SIZE

    def get_batch(self, idx, batch_size):
        try:
            subset = self.sentences.ids[idx: min(idx + batch_size + self.seq_size + 1, len(self.sentences) - 1)]
            x, y = [], []
            for i in range(batch_size):
                x.append(torch.tensor(subset[i:i+self.seq_size], device=device))
                y.append(torch.tensor(subset[i+1:i+self.seq_size+1], device=device))
            return torch.stack(x, dim=0), torch.stack(y, dim=0)
        except:
            return -1, -1

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return (
            torch.tensor(self.sentences.ids[index:index + self.seq_size], device=device),
            torch.tensor(self.sentences.ids[index + 1:index + self.seq_size + 1], device=device)
        )


if __name__ == "__main__":
    preprocessing_pipeline(save=True)
