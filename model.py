import math
from numpy import dtype

# Models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab, vocab_size, class_num=3, dimension=128, embed_dim=512, multiple_fc=4, dropout=0.5):
        super(LSTM, self).__init__()

        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.dimension = dimension
        
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.bat1 = nn.BatchNorm1d(multiple_fc * dimension)

        self.fc = nn.Linear(2 * dimension, multiple_fc * dimension) # bidirectional lstm input shape (x 2)
        self.fc2 = nn.Linear(multiple_fc * dimension, class_num)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = out_reduced

        text_out = self.drop(self.relu(self.fc(text_fea)))
        text_out = self.fc2(text_out)

        return text_out


class CNN1d(nn.Module):
    def __init__(self, vocab, vocab_size, embed_dim=512, n_filters=128, 
        multiple_fc=4, class_num=3, dropout=0.4, kernel_sizes=[2, 3]):
        
        super().__init__()
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.max_kernerl_size = max(kernel_sizes)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embed_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = k)
                                    for k in kernel_sizes
                                    ])
        
        self.relu = nn.ReLU()
        self.fc = nn.Linear(len(kernel_sizes) * n_filters, len(kernel_sizes) * n_filters * multiple_fc)
        self.fc2 = nn.Linear(len(kernel_sizes) * n_filters * multiple_fc, class_num)
        
        self.drop = nn.Dropout(p=dropout)
        
    def forward(self, text, text_len):
        
        #text_shape = [batch size, sent len]
        embedded = self.embedding(text)   
        #embedded_shape = [batch size, sent len, emb dim]

        # pad for CNN kernel
        if embedded.shape[1] < self.max_kernerl_size:
            embedded = F.pad(embedded, (0, 0, 0, self.max_kernerl_size - embedded.shape[1]), value=0)
        
        embedded = embedded.permute(0, 2, 1)
        #embedded_shape = [batch size, emb dim, sent len]
        
        conved = [torch.functional.F.relu(conv(embedded)) for conv in self.convs]
        #conved_n_shape = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [torch.functional.F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n_shape = [batch size, n_filters]
        
        cat = self.drop(self.relu(torch.cat(pooled, dim = 1)))
        #cat_shape = [batch size, n_filters * len(filter_sizes)]

        out = self.drop(self.relu(self.fc(cat)))
        out = self.fc2(out)
        
        return out


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class Combination(nn.Module):
    def __init__(self, vocab, vocab_size, class_num=3, embed_dim=512, hidden_size=256, 
                n_filters=128, d_prob=0.4, emb_vectors=None, mode="static", kernel_sizes=[2, 3], spatial_drop=0.1):
        super(Combination, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding_dim = embed_dim
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.num_filters = n_filters
        self.num_classes = class_num
        self.hidden_dim = n_filters
        self.max_kernerl_size = max(kernel_sizes)
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=1)
        self.embedding_dropout = SpatialDropout(spatial_drop)
        self.relu = nn.ReLU()

        if emb_vectors is not None:
            self.load_embeddings(emb_vectors)

        self.conv = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_dim,
                                             out_channels=n_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.lstm = nn.LSTM(input_size=self.hidden_dim * len(kernel_sizes), hidden_size=hidden_size,
                             bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(hidden_size * 2, self.hidden_dim)
        self.fc_final = nn.Linear(self.hidden_dim, class_num)

    def forward(self, x, x_len):
        x_emb = self.embedding(x)
        x_emb = self.embedding_dropout(x_emb)
        
        # pad for CNN kernel
        if x_emb.shape[1] < self.max_kernerl_size:
            x_emb = F.pad(x_emb, (0, 0, 0, self.max_kernerl_size - x_emb.shape[1]), value=0)
            
        x = [self.relu(conv(x_emb.transpose(1, 2))) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = x.unsqueeze(0)

        h_lstm, _ = self.lstm(x) # bidirectional lstm input shape (x 2)

        out = h_lstm.squeeze(0)
        out = self.dropout(self.relu(self.fc(out)))
        out = self.fc_final(out)

        return out