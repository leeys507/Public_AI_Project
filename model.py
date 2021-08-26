import math
from numpy import dtype

# Models
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab_size, class_num=3, dimension=128, embed_dim=300, use_embed=False, pre_embed=None):
        super(LSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dimension = dimension

        # if use_embed and pre_embed is not None:
        #     self.embedding.weight.data.copy_(torch.from_numpy(pre_embed))
        
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, class_num)


    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_out = self.fc(text_fea)
        #text_out = torch.sigmoid(text_fea) # BCE Loss

        return text_out


class CNN1d(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, n_filters=100, class_num=3, dropout=0.25, filter_sizes=[1]):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embed_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, class_num)
        
        self.dropout = nn.Dropout(dropout)
        
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
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat_shape = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)