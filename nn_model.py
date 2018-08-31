import torch
import torch.nn as nn
import torch.nn.functional as torchF


class ReorderingEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(ReorderingEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.e2h = nn.Linear(embed_size, hidden_size)
        self.forward_gru = nn.GRU(hidden_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.h2h = nn.Linear(hidden_size * 2, hidden_size)
        self.h2v = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, xs, fhidden):
        embedded = torch.tanh(self.e2h(self.embedding(xs)))
        fout, fh = self.forward_gru(embedded, fhidden)
        preds = self.h2v(torch.tanh(fout))
        return preds, fh

    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.e2h = nn.Linear(embed_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.c2h = nn.Linear(hidden_size * 2, hidden_size)
        self.h2e = nn.Linear(hidden_size, embed_size)
        self.e2v = nn.Linear(embed_size, vocab_size)

    def forward(self, xs, hidden, ehs):
        batch_size = xs.size(0)
        embedded = torch.tanh(self.e2h(self.embedding(xs)))
        out, h = self.gru(embedded, hidden)
        attention_weight = torch.exp(torch.tanh(torch.sum(ehs * h, dim=2)))
        attention = attention_weight / torch.sum(attention_weight, dim=0)
        context_vector = torch.sum(ehs * attention.view(-1, batch_size, 1), dim=0)
        output = torch.tanh(self.c2h(torch.cat((h.squeeze(0), context_vector), dim=1)))
        output = self.e2v(torch.tanh(self.h2e(output)))
        return output, h

    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttentionSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(AttentionSeq2Seq, self).__init__()
        self.enc = ReorderingEncoder(vocab_size, embed_size, hidden_size)
        self.dec = AttentionDecoder(vocab_size, embed_size, hidden_size)

    def forward(self, xs, ys):
        pass
