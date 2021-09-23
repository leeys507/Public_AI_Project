import math
from numpy import dtype

# Models
import torch
import torch.nn.functional as F
import torch.nn as nn


class Combination(nn.Module):
    def __init__(self, vocab, vocab_size, class_num=3, embed_dim=512, hidden_size=256, 
                n_filters=128, d_prob=0.4, kernel_sizes=[2, 3], spatial_drop=0.1):
        super(Combination, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding_dim = embed_dim

        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.num_filters = n_filters
        self.max_kernerl_size = max(kernel_sizes)
        self.d_prob = d_prob

        self.num_classes = class_num
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=1)
        self.embedding_dropout = SpatialDropout(p=spatial_drop)
        self.relu = nn.ReLU()

        self.conv = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_dim,
                                             out_channels=n_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.lstm = nn.LSTM(input_size=self.num_filters * len(kernel_sizes), hidden_size=hidden_size,
                             bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(p=d_prob)
        self.fc = nn.Linear(hidden_size * 2, self.num_filters) # bidirectional lstm input shape (x 2)
        self.fc_final = nn.Linear(self.num_filters, class_num)

    def forward(self, x, x_len):
        x_emb = self.embedding(x)
        x_emb = self.embedding_dropout(x_emb)
        
        # pad for CNN kernel
        if x_emb.shape[1] < self.max_kernerl_size:
            x_emb = F.pad(x_emb, (0, 0, 0, self.max_kernerl_size - x_emb.shape[1]), value=0)
            
        x = [self.relu(conv(x_emb.transpose(1, 2))) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = x.unsqueeze(0)

        h_lstm, _ = self.lstm(x)

        out = h_lstm.squeeze(0)
        out = self.dropout(self.relu(self.fc(out)))
        out = self.fc_final(out)

        return out


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x