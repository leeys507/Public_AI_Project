import math
from numpy import dtype

# Models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab, vocab_size, class_num=3, dimension=128, embed_dim=300, dropout=0.4):
        super(LSTM, self).__init__()

        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dimension = dimension
        
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(2 * dimension, 3 * dimension)
        self.fc2 = nn.Linear(3 * dimension, dimension)
        self.fc3 = nn.Linear(dimension, class_num)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_out = self.drop(self.relu(self.fc(text_fea)))

        text_out = self.drop(self.relu(self.fc2(text_out)))
        text_out = self.fc3(text_out)

        return text_out


class CNN1d(nn.Module):
    def __init__(self, vocab, vocab_size, embed_dim=300, n_filters=128, multiple_fc=4, class_num=3, dropout=0.4, kernel_sizes=[1]):
        
        super().__init__()
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embed_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in kernel_sizes
                                    ])
        
        self.relu = nn.ReLU()
        self.fc = nn.Linear(len(kernel_sizes) * n_filters, len(kernel_sizes) * n_filters * multiple_fc)
        self.fc2 = nn.Linear(len(kernel_sizes) * n_filters * multiple_fc, class_num)
        
        self.drop = nn.Dropout(p=dropout)
        
    def forward(self, text, text_len):
        
        #text_shape = [batch size, sent len]
        embedded = self.embedding(text)   
        #embedded_shape = [batch size, sent len, emb dim]
        
        embedded = embedded.permute(0, 2, 1)
        #embedded_shape = [batch size, emb dim, sent len]
        
        conved = [torch.functional.F.relu(conv(embedded)) for conv in self.convs]
        #conved_n_shape = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [torch.functional.F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n_shape = [batch size, n_filters]
        
        cat = self.drop(torch.cat(pooled, dim = 1))
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
    def __init__(self, vocab, vocab_size, class_num=3, embed_dim=300, hidden_dim=128, lstm_units=64, 
                n_filters=100, d_prob=0.25, emb_vectors=None, mode="static", kernel_sizes=[1], spatial_drop=0.1):
        super(Combination, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embedding_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = n_filters
        self.num_classes = class_num
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.embedding_dropout = SpatialDropout(spatial_drop)

        if emb_vectors is not None:
            self.load_embeddings(emb_vectors)

        self.conv = nn.ModuleList([nn.Conv1d(in_channels=embed_dim,
                                             out_channels=n_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.lstm1 = nn.LSTM(embed_dim, hidden_size=lstm_units,
                             bidirectional=True, batch_first=True)
        # self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units,
        #                      bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(len(kernel_sizes) * n_filters, hidden_dim)
        self.fc_total = nn.Linear(hidden_dim * 1 + lstm_units * 4, hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, class_num)

    def forward(self, x, x_len):
        x_emb = self.embedding(x)
        x_emb = self.embedding_dropout(x_emb)
        
        # pad for CNN kernel 5
        if x_emb.shape[1] < 5:
            x_emb = F.pad(x_emb, (0, 0, 0, 5 - x_emb.shape[1]), value=0)
            
        x = [F.relu(conv(x_emb.transpose(1, 2))) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))

        h_lstm1, _ = self.lstm1(x_emb)
        #h_lstm2, _ = self.lstm2(h_lstm1)

        # average pooling
        avg_pool2 = torch.mean(h_lstm1, 1)
        # global max pooling
        max_pool2, _ = torch.max(h_lstm1, 1)


        out = torch.cat([x, avg_pool2, max_pool2], dim=1)
        out = F.relu(self.fc_total(self.dropout(out)))
        out = self.fc_final(out)

        return out

    def load_embeddings(self, emb_vectors):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(emb_vectors)
            if 'non' not in self.mode:
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')
            else:
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')
        elif self.mode == 'rand':
            print('Randomly initialized embeddings are used.')
        else:
            raise ValueError(
                'Unexpected value of mode. Please choose from static, nonstatic, rand.')