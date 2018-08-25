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
        while k < len(indexes):
            encoder_optimizer.zero_grad(), decoder_optimizer.zero_grad()
            batch_idx = indexes[k: min(k + batch_size, len(indexes))]
            enc_loss, dec_loss = 0, 0
            batch_t_s, batch_t_r, batch_t_t = [], [], []
            init_hidden = encoder.initHidden(len(batch_idx), device)
            pred_seq = [[] for _ in range(len(batch_idx))]
            for idx in batch_idx:
                batch_t_s.append(train_src[idx])
                batch_t_r.append(train_re[idx])
                batch_t_t.append(train_tgt[idx])
            max_s_len = max(len(_t_s) for _t_s in batch_t_s)
            for i in range(len(batch_t_s)):
                batch_t_s[i] = batch_t_s[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_t_s[i]))
                batch_t_r[i] = batch_t_r[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_t_r[i]))
            max_t_len = max(len(_t_t) for _t_t in batch_t_t)
            for i in range(len(batch_t_t)):
                batch_t_t[i] = batch_t_t[i] + [s_vocab['<EOS>']] * (max_t_len - len(batch_t_t[i]) + 1)

            xs = torch.tensor(batch_t_s).to(device)
            batch_t_r = torch.tensor(batch_t_r)
            pred_dists, ehs = encoder(xs, init_hidden)
            for i in range(pred_dists.size(1)):
                enc_loss += encoder_criterion(pred_dists[:, i, :], torch.tensor(batch_t_r[:, i]).to(device))

            ys = torch.tensor(batch_t_t)
            dhidden = decoder.initHidden(len(batch_idx), device)
            pred_words = torch.tensor([[t_vocab['<BOS>']] for _ in range(len(batch_idx))]).to(device)
            for j in range(max_t_len):
                preds, dhidden = decoder(pred_words, dhidden, ehs)
                topv, topi = preds.topk(1)
                dec_loss += decoder_criterion(preds, torch.tensor(ys[:, j]).to(device))
                pred_words = torch.tensor(ys[:, j + 1]).view(-1, 1).to(device)
                for i in range(len(pred_seq)):
                    pred_seq[i].append(topi[i].item())
            i = random.randrange(0, len(batch_idx))
            print(' '.join(t_vocab_list[t] for t in batch_t_t[i]))
            print(' '.join(t_vocab_list[t] if t < len(t_vocab_list) else '<UNK>' for t in pred_seq[i]))

            enc_sum_loss += enc_loss.item()
            dec_sum_loss += dec_loss.item()
            dec_loss.backward(retain_graph=True)
            if train_reorder:
                enc_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
            encoder_optimizer.step()
            decoder_optimizer.step()
            print('\r%d sentences learned, enc loss: %.6f, dec loss: %.6f'
                  % (k + len(batch_idx), enc_loss.item(), dec_loss.item()), end='')
            k += batch_size

        print('\nencoder loss:', enc_sum_loss, 'decoder loss:', dec_sum_loss)
        train_losses.append(dec_sum_loss)
        enc_dev_loss, dec_dev_loss = evaluate(encoder, decoder, valid_src, valid_re, valid_tgt, 30, device, s_vocab, t_vocab)
        print('dev enc loss:', enc_dev_loss.item(), 'dev dec loss:', dec_dev_loss.item())

    return train_losses


def evaluate(encoder, decoder, src, src_re, tgt, eval_len, device, s_vocab, t_vocab):
    k = 0
    encoder_criterion, decoder_criterion = torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()
    enc_dev_loss, dec_dev_loss = 0, 0
    while k < len(src):
        batch_src, batch_re, batch_tgt = [], [], []
        for i in range(k, min(k + eval_len, len(src))):
            batch_src.append(src[i])
            batch_re.append(src_re[i])
            batch_tgt.append(tgt[i])
        max_s_len = max(len(s) for s in batch_src)
        for i in range(len(batch_src)):
            batch_src[i] = batch_src[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_src[i]))
            batch_re[i] = batch_re[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_re[i]))
        max_t_len = max(len(t) for t in batch_tgt)
        for i in range(len(batch_tgt)):
            batch_tgt[i] = batch_tgt[i] + [t_vocab['<EOS>']] * (max_t_len - len(batch_tgt[i]) + 1)
        xs = torch.tensor(batch_src).to(device)
        reorder = torch.tensor(batch_re)
        init_hidden = encoder.initHidden(len(batch_src), device)
        pred_dists, ehs = encoder(xs, init_hidden)
        for i in range(pred_dists.size(1)):
            enc_dev_loss += encoder_criterion(pred_dists[:, i, :], torch.tensor(reorder[:, i]).to(device))

        ys = torch.tensor(batch_tgt)
        dhidden = decoder.initHidden(len(batch_src), device)
        pred_words = torch.tensor([[t_vocab['<BOS>']] for _ in range(len(batch_src))]).to(device)
        for j in range(max_t_len):
            preds, dhidden = decoder(pred_words, dhidden, ehs)
            dec_dev_loss += decoder_criterion(preds, torch.tensor(ys[:, j])).to(device)
            pred_words = torch.tensor(ys[:, j + 1]).view(-1, 1).to(device)
        k += eval_len
    return enc_dev_loss, dec_dev_loss


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
