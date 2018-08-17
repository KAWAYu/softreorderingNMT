#!/usr/bin/python
# -*- coding; utf-8 -*-

from io import open


def make_vocab(filepath, max_size, init_vocab={}):
    vocab = {'<UNK>': 0, '<EOS>': 1, '<BOS>': 2}
    vocab_count = {}
    with open(filepath, encoding='utf-8') as fin:
        for line in fin:
            tokens = line.strip().split(' ')
            for token in tokens:
                if token in vocab_count:
                    vocab_count[token] += 1
                else:
                    vocab_count[token] = 1
    for k, _ in sorted(vocab_count.items(), key=lambda x: -x[1]):
        vocab[k] = len(vocab)
        if len(vocab) >= max_size:
            break
    return vocab


def batch_generation(seq1, seq2, shuffle=True):
    pass


if __name__ == '__main__':
    print('This is a utility box.')
