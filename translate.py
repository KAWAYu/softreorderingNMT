#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
from io import open
import pickle
import torch
import torch.nn as nn
import random

from nn_model import ReorderingEncoder, AttentionDecoder

device = torch.device('cpu')
torch.set_grad_enabled(False)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-s')
    parser.add_argument('--output', '-o')
    parser.add_argument('--encoder_model', '-em', default='encoder.model')
    parser.add_argument('--decoder_model', '-dm', default='decoder.model')
    parser.add_argument('--s_vocab', '-sv', default='s_vocab.pkl')
    parser.add_argument('--t_vocab', '-tv', default='t_vocab.pkl')
    parser.add_argument('--vocab_size', '-vs', default=50000, type=int)
    parser.add_argument('--embed_size', '-es', default=500, type=int)
    parser.add_argument('--hidden_size', '-hs', default=500, type=int)
    parser.add_argument('--max_len', '-l', default=100, type=int)
    parser.add_argument('--gpu_id', '-g', default=-1, type=int)
    return parser.parse_args()


def translate(output, src, encoder, decoder, s_vocab, t_vocab, vocab_list, max_len, device):
    batch_len = 30
    k = 0
    with open(output, 'w') as fout:
        while k < len(src):
            src_batch = src[k: min(k + batch_len, len(src))]
            max_s_len = max(len(s) + 1 for s in src_batch)
            for i in range(len(src_batch)):
                src_batch[i] = src_batch[i] + [s_vocab['<EOS>']] * (max_s_len - len(src_batch[i]))
            xs = torch.tensor(src_batch, device=device).t().contiguous()
            enc_init_hidden = encoder.initHidden(len(src_batch), device)
            _, ehs = encoder(xs, enc_init_hidden)

            prev_words = torch.tensor([[t_vocab['<BOS>'] for _ in range(len(src_batch))]], device=device)
            dhidden = decoder.initHidden(len(src_batch), device)
            pred_seqs = [[] for _ in range(len(src_batch))]
            for _ in range(max_len):
                preds, dhidden = decoder(prev_words, dhidden, ehs)
                _, topi = preds.topk(1)
                for i in range(len(pred_seqs)):
                    pred_seqs[i].append(topi[i].item())
                if all(topii == t_vocab['<EOS>'] for topii in topi):
                    break
                prev_words = topi.view(1, -1).detach()

            for pred_seq in pred_seqs:
                _pred_seq = []
                for p in pred_seq:
                    if p == t_vocab['<EOS>']:
                        break
                    _pred_seq.append(p)
                print(' '.join(vocab_list[t] if 0 <= t < len(vocab_list) else vocab_list[0]
                               for t in _pred_seq), file=fout)
            k += batch_len


def main():
    global device
    args = parse()
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu_id))

    s_vocab = pickle.load(open(args.s_vocab, 'rb'))
    t_vocab = pickle.load(open(args.t_vocab, 'rb'))
    t_vocab_list = [k for k, _ in sorted(t_vocab.items(), key=lambda x: x[1])]

    encoder = ReorderingEncoder(args.vocab_size, args.embed_size, args.hidden_size)
    decoder = AttentionDecoder(args.vocab_size, args.embed_size, args.hidden_size)
    encoder.load_state_dict(torch.load(args.encoder_model))
    decoder.load_state_dict(torch.load(args.decoder_model))
    encoder.to(device), decoder.to(device)
    src = []
    with open(args.src, encoding='utf-8') as fin:
        for line in fin:
            src.append([s_vocab[t] if t in s_vocab else s_vocab['<UNK>'] for t in line.strip().split(' ')])
    translate(args.output, src, encoder, decoder, s_vocab, t_vocab, t_vocab_list, args.max_len, device)


if __name__ == '__main__':
    main()
