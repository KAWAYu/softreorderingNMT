#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from io import open
import torch
import random
import matplotlib
import numpy as np
import os
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import my_utils as utils
import nn_model

device = torch.device('cpu')


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
    parser.add_argument('--datapath', '-dp', default='data', type=str)
    parser.add_argument('--train_baseline', '-tbase', default=False, action='store_true')

    args = parser.parse_args()
    return args


def train(train_src, train_re, train_tgt, valid_src, valid_re, valid_tgt,
          s_vocab, t_vocab, encoder, decoder, epoch_num, batch_size, device, train_reorder=True, verbose=2,
          data_path=None):
    t_vocab_list = [k for k, _ in sorted(t_vocab.items(), key=lambda x: x[1])]
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), weight_decay=1e-4)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), weight_decay=1e-4)
    encoder_criterion, decoder_criterion = torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()
    train_enc_losses, train_dec_losses, valid_enc_losses, valid_dec_losses = [], [], [], []

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
            for idx in batch_idx:
                batch_t_s.append(train_src[idx])
                batch_t_r.append(train_re[idx])
                batch_t_t.append(train_tgt[idx])
            max_s_len = max(len(_t_s) + 1 for _t_s in batch_t_s)
            for i in range(len(batch_t_s)):
                batch_t_s[i] = batch_t_s[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_t_s[i]))
                batch_t_r[i] = batch_t_r[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_t_r[i]))
            max_t_len = max(len(_t_t) + 1 for _t_t in batch_t_t)
            for i in range(len(batch_t_t)):
                batch_t_t[i] = batch_t_t[i] + [t_vocab['<EOS>']] * (max_t_len - len(batch_t_t[i]))

            xs = torch.tensor(batch_t_s).to(device).t().contiguous()
            batch_t_r = torch.tensor(batch_t_r)
            init_hidden = encoder.initHidden(len(batch_idx), device)
            pred_dists, ehs = encoder(xs, init_hidden)
            for i in range(batch_t_r.size(1)):
                enc_loss += encoder_criterion(pred_dists[i, :, :], torch.tensor(batch_t_r[:, i]).to(device))

            if train_reorder:
                enc_loss.backward(retain_graph=True)

            ys = torch.tensor(batch_t_t)
            dhidden = decoder.initHidden(len(batch_idx), device)
            pred_words = torch.tensor([[t_vocab['<BOS>'] for _ in range(len(batch_idx))]]).to(device)
            pred_seqs = [[] for _ in range(len(batch_idx))]
            for j in range(max_t_len - 1):
                preds, dhidden = decoder(pred_words, dhidden, ehs)
                dec_loss += decoder_criterion(preds, torch.tensor(ys[:, j]).to(device))
                pred_words = torch.tensor(ys[:, j]).unsqueeze(0).to(device)
                _, topi = preds.topk(1)
                for i in range(len(pred_seqs)):
                    pred_seqs[i].append(topi[i].item())
            if verbose > 1:
                idx = random.choice([j for j in range(len(batch_idx))])
                print('\n' + ' '.join(t_vocab_list[i] if 0 < i < len(t_vocab_list) else t_vocab_list[0]
                                      for i in batch_t_t[idx]))
                print(' '.join(t_vocab_list[i] if 0 < i < len(t_vocab_list) else t_vocab_list[0]
                               for i in pred_seqs[idx]))

            enc_sum_loss += enc_loss.item()
            dec_sum_loss += dec_loss.item()
            dec_loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
            encoder_optimizer.step()
            decoder_optimizer.step()
            if verbose > 0:
                print('\r%d sentences learned, enc loss: %.6f, dec loss: %.6f'
                      % (k + len(batch_idx), enc_loss.item(), dec_loss.item()), end='')
            k += batch_size

        print('\nencoder loss:', enc_sum_loss / len(train_src), 'decoder loss:', dec_sum_loss / len(train_src))

        train_enc_losses.append(enc_sum_loss / len(train_src))
        train_dec_losses.append(dec_sum_loss / len(train_src))
        encoder_optimizer.zero_grad(), decoder_optimizer.zero_grad()
        enc_dev_loss, dec_dev_loss = evaluate(
            encoder, decoder, valid_src, valid_re, valid_tgt, 100, device, s_vocab, t_vocab)
        valid_enc_losses.append(enc_dev_loss.item() / len(valid_src))
        valid_dec_losses.append(dec_dev_loss.item() / len(valid_src))
        print('dev enc loss:', enc_dev_loss.item() / len(valid_src),
              'dev dec loss:', dec_dev_loss.item() / len(valid_tgt))

        if data_path is not None:
            torch.save(encoder.state_dict(), data_path + '.epoch%d.enc' % (e + 1))
            torch.save(decoder.state_dict(), data_path + '.epoch%d.dec' % (e + 1))

    return train_enc_losses, train_dec_losses, valid_enc_losses, valid_dec_losses


def evaluate(encoder, decoder, src, src_re, tgt, eval_len, device, s_vocab, t_vocab):
    t_vocab_list = [k for k, _ in sorted(t_vocab.items(), key=lambda x: x[1])]
    with torch.set_grad_enabled(False):
        k = 0
        encoder_criterion, decoder_criterion = torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()
        enc_dev_loss, dec_dev_loss = 0, 0
        while k < len(src):
            batch_src, batch_re, batch_tgt = [], [], []
            for i in range(k, min(k + eval_len, len(src))):
                batch_src.append(src[i])
                batch_re.append(src_re[i])
                batch_tgt.append(tgt[i])
            max_s_len = max(len(s) + 1 for s in batch_src)
            for i in range(len(batch_src)):
                batch_src[i] = batch_src[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_src[i]))
                batch_re[i] = batch_re[i] + [s_vocab['<EOS>']] * (max_s_len - len(batch_re[i]))
            max_t_len = max(len(t) + 1 for t in batch_tgt)
            for i in range(len(batch_tgt)):
                batch_tgt[i] = batch_tgt[i] + [t_vocab['<EOS>']] * (max_t_len - len(batch_tgt[i]))

            xs = torch.tensor(batch_src, device=device).to(device).t().contiguous()
            reorder = torch.tensor(batch_re)
            init_hidden = encoder.initHidden(len(batch_src), device)
            pred_dists, ehs = encoder(xs, init_hidden)
            for i in range(reorder.size(1)):
                enc_dev_loss += encoder_criterion(pred_dists[i, :, :], torch.tensor(reorder[:, i], device=device))

            ys = torch.tensor(batch_tgt)
            dhidden = decoder.initHidden(len(batch_src), device)
            pred_words = torch.tensor([[t_vocab['<BOS>'] for _ in range(len(batch_src))]], device=device)
            i = random.choice([j for j in range(len(batch_src))])
            pred_seq = []
            for j in range(max_t_len - 1):
                preds, dhidden = decoder(pred_words, dhidden, ehs)
                _, topi = preds.topk(1)
                dec_dev_loss += decoder_criterion(preds, torch.tensor(ys[:, j], device=device))
                pred_seq.append(topi[i].item())
                pred_words = topi.view(1, -1).detach()
            print(' '.join(t_vocab_list[i] for i in batch_tgt[i]))
            print(' '.join(t_vocab_list[i] if 0 < i < len(t_vocab_list) else t_vocab_list[0] for i in pred_seq))
            k += eval_len
        return enc_dev_loss, dec_dev_loss


def main():
    global device
    args = parse()
    s_vocab = utils.make_vocab(args.train_src, args.vocab_size)
    t_vocab = utils.make_vocab(args.train_tgt, args.vocab_size)
    train_source_sentences, train_target_sentences, train_reorder_sentences = [], [], []
    valid_source_sentences, valid_target_sentences, valid_reorder_sentences = [], [], []
    if not os.path.isdir(args.datapath):
        os.mkdir(args.datapath)

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
    train_enc_loss_w_re, train_dec_loss_w_re, valid_enc_loss_w_re, valid_dec_loss_w_re = train(
        train_source_sentences, train_reorder_sentences, train_target_sentences,
        valid_source_sentences, valid_reorder_sentences, valid_target_sentences,
        s_vocab, t_vocab, encoder, decoder, args.epochs, args.batch_size, device, verbose=1,
        data_path=os.path.join(args.datapath, 'with_reorder.model'))

    if args.train_baseline:
        encoder = nn_model.ReorderingEncoder(args.vocab_size, args.embed_size, args.hidden_size).to(device)
        decoder = nn_model.AttentionDecoder(args.vocab_size, args.embed_size, args.hidden_size).to(device)
        train_enc_loss_wo_re, train_dec_loss_wo_re, valid_enc_loss_wo_re, valid_dec_loss_wo_re = train(
            train_source_sentences, train_reorder_sentences, train_target_sentences,
            valid_source_sentences, valid_reorder_sentences, valid_target_sentences,
            s_vocab, t_vocab, encoder, decoder, args.epochs, args.batch_size, device, False, verbose=1,
            data_path=os.path.join(args.datapath, 'without_reorder.model'))

    pickle.dump(s_vocab, open(os.path.join(args.datapath, 's_vocab.pkl'), 'wb'))
    pickle.dump(t_vocab, open(os.path.join(args.datapath, 't_vocab.pkl'), 'wb'))

    plt.plot(np.array([i for i in range(args.epochs)]), train_enc_loss_w_re, 'b-',
             label='train encoder loss with reorder')
    plt.plot(np.array([i for i in range(args.epochs)]), train_dec_loss_w_re, 'b:',
             label='train decoder loss with reorder')
    plt.plot(np.array([i for i in range(args.epochs)]), valid_enc_loss_w_re, 'g-',
             label='valid encoder loss with reorder')
    plt.plot(np.array([i for i in range(args.epochs)]), valid_dec_loss_w_re, 'g:',
             label='valid decoder loss with reorder')
    if args.train_baseline:
        plt.plot(np.array([i for i in range(args.epochs)]), train_enc_loss_wo_re, 'r-',
                 label='train encoder loss without reorder')
        plt.plot(np.array([i for i in range(args.epochs)]), train_dec_loss_wo_re, 'r:',
                 label='train decoder loss without reorder')
        plt.plot(np.array([i for i in range(args.epochs)]), valid_enc_loss_wo_re, 'y-',
                 label='valid encoder loss without reorder')
        plt.plot(np.array([i for i in range(args.epochs)]), valid_dec_loss_wo_re, 'y:',
                 label='valid decoder loss without reorder')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curve.pdf')


if __name__ == '__main__':
    main()
