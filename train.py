#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from io import open
import torch
import random
import matplotlib
import numpy as np
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import my_utils as utils
import nn_model

device = torch.device('cpu')


def train(train_src, train_re, train_tgt, valid_src, valid_re, valid_tgt,
          s_vocab, t_vocab, encoder, decoder, epoch_num, batch_size, device, train_reorder=True):
    t_vocab_list = [k for k, _ in sorted(t_vocab.items(), key=lambda x: x[1])]
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), weight_decay=1e-6)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), weight_decay=1e-6)
    encoder_criterion, decoder_criterion = torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()
    train_losses = []

    for e in range(epoch_num):
        print('Epoch %d begin...' % (e + 1))
        enc_sum_loss, dec_sum_loss = 0, 0
        indexes = [i for i in range(len(train_src))]
        random.shuffle(indexes)
        k = 0
        for idx in indexes:
            encoder_optimizer.zero_grad(), decoder_optimizer.zero_grad()
            if (k + 1) % 10 == 0:
                print('\r%d data end' % (k + 1))
            enc_loss, dec_loss = 0, 0
            t_s, t_r, t_t = train_src[idx], train_re[idx], train_tgt[idx]
            ehs = torch.zeros(len(t_s), encoder.hidden_size)
            init_hidden = encoder.initHidden(1)
            xs = torch.tensor([[t] for t in t_s]).to(device)
            pred_dists, ehs = encoder(xs, init_hidden)
            for pred_dist, t in zip(pred_dists, t_r):
                enc_loss += encoder_criterion(pred_dist, torch.tensor([t]))
            # print(pred_dists.size())
            # enc_loss = encoder_criterion(pred_dist, torch.tensor([[t] for t in t_r]))
            # for i in range(len(t_s)):
            #     fxs, bxs = torch.tensor([t_s[i]]), torch.tensor([t_s[-i - 1]])
            #     pred_dists, fhbh, fhidden, bhidden = encoder(fxs, bxs, fhidden, bhidden)
            #     enc_loss += encoder_criterion(pred_dists.squeeze(1), torch.tensor([t_r[i]]))
            #     ehs[i] = fhbh[0, 0]

            pred_words = torch.tensor([[t_vocab['<BOS>']]]).to(device)
            pred_seq = []
            dhidden = decoder.initHidden(1)
            for j in range(len(t_t)):
                preds, dhidden = decoder(pred_words, dhidden, ehs)
                topv, topi = preds.topk(1)
                dec_loss += decoder_criterion(preds, torch.tensor([t_t[j]]))
                pred_words = torch.tensor([[t_t[j]]]).to(device)
                pred_seq.append(topi.item())
            if (k + 1) % 10 == 0:
                print(' '.join(t_vocab_list[t] if t < len(t_vocab_list) else t_vocab_list[0] for t in t_t))
                print(' '.join(t_vocab_list[t] if t < len(t_vocab_list) else t_vocab_list[0]for t in pred_seq))

            enc_sum_loss += enc_loss.item()
            dec_sum_loss += dec_loss.item()
            dec_loss.backward(retain_graph=True)
            if train_reorder:
                enc_loss.backward(retain_graph=True)
            encoder_optimizer.step()
            decoder_optimizer.step()
            k += 1
        train_losses.append(dec_sum_loss)
        print('encoder loss:', enc_sum_loss, 'decoder loss:', dec_sum_loss)
    return train_losses


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train_src')
    parser.add_argument('-train_re')
    parser.add_argument('-train_tgt')
    parser.add_argument('-valid_src')
    parser.add_argument('-valid_re')
    parser.add_argument('-valid_tgt')
    parser.add_argument('--vocab_size', '-vs', default=50000, type=int)
    parser.add_argument('--embed_size', '-es', default=500, type=int)
    parser.add_argument('--hidden_size', '-hs', default=500, type=int)
    parser.add_argument('--batch_size', '-bs', default=500, type=int)
    parser.add_argument('--epochs', '-e', default=20, type=int)
    parser.add_argument('--gpu_id', '-g', default=-1, type=int)

    args = parser.parse_args()
    return args


def main():
    global device
    args = parse()
    s_vocab = utils.make_vocab(args.train_src, args.vocab_size)
    t_vocab = utils.make_vocab(args.train_tgt, args.vocab_size)
    train_source_sentences, train_target_sentences, train_reorder_sentences = [], [], []
    valid_source_sentences, valid_target_sentences, valid_reorder_sentences = [], [], []

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu_id))

    with open(args.train_src, encoding='utf-8') as fin:
        for line in fin:
            train_source_sentences.append([
                s_vocab[token] if token in s_vocab else s_vocab['<UNK>'] for token in line.strip().split(' ')
            ])
    with open(args.train_re, encoding='utf-8') as fin:
        for line in fin:
            train_reorder_sentences.append([
                s_vocab[token] if token in s_vocab else s_vocab['<UNK>'] for token in line.strip().split(' ')
            ])
    with open(args.train_tgt, encoding='utf-8') as fin:
        for line in fin:
            train_target_sentences.append([
                t_vocab[token] if token in t_vocab else t_vocab['<UNK>'] for token in line.strip().split(' ')
            ])
    with open(args.valid_src, encoding='utf-8') as fin:
        for line in fin:
            valid_source_sentences.append([
                s_vocab[token] if token in s_vocab else s_vocab['<UNK>'] for token in line.strip().split(' ')
            ])
    with open(args.valid_re, encoding='utf-8') as fin:
        for line in fin:
            valid_reorder_sentences.append([
                s_vocab[token] if token in s_vocab else s_vocab['<UNK>'] for token in line.strip().split(' ')
            ])
    with open(args.valid_tgt, encoding='utf-8') as fin:
        for line in fin:
            valid_target_sentences.append([
                t_vocab[token] if token in t_vocab else t_vocab['<UNK>'] for token in line.strip().split(' ')
            ])

    encoder = nn_model.ReorderingEncoder(args.vocab_size, args.embed_size, args.hidden_size).to(device)
    decoder = nn_model.AttentionDecoder(args.vocab_size, args.embed_size, args.hidden_size).to(device)
    train_loss_with_reorder = train(train_source_sentences, train_reorder_sentences, train_target_sentences,
                       valid_source_sentences, valid_reorder_sentences, valid_target_sentences,
                       s_vocab, t_vocab, encoder, decoder, args.epochs, args.batch_size, device)
    torch.save(encoder.state_dict(), 'encoder.model')
    torch.save(decoder.state_dict(), 'decoder.model')
    encoder = nn_model.ReorderingEncoder(args.vocab_size, args.embed_size, args.hidden_size).to(device)
    decoder = nn_model.AttentionDecoder(args.vocab_size, args.embed_size, args.hidden_size).to(device)
    train_loss_wo_reorder = train(train_source_sentences, train_reorder_sentences, train_target_sentences,
                                    valid_source_sentences, valid_reorder_sentences, valid_target_sentences,
                                    s_vocab, t_vocab, encoder, decoder, args.epochs, args.batch_size, device, False)
    torch.save(encoder.state_dict(), 'encoder_base.model')
    torch.save(decoder.state_dict(), 'decoder_base.model')

    pickle.dump(s_vocab, open('s_vocab.pkl', 'wb'))
    pickle.dump(t_vocab, open('t_vocab.pkl', 'wb'))

    plt.plot(np.array([i for i in range(args.epochs)]), train_loss_with_reorder, label='train loss with reorder')
    plt.plot(np.array([i for i in range(args.epochs)]), train_loss_wo_reorder, label='train loss w/o reorder')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('loss_curve.pdf')


if __name__ == '__main__':
    main()
