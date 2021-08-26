import torch
# from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, ReversibleField

import pandas as pd
from sklearn.model_selection import train_test_split


def create_split_csv(raw_data_path=".", dest_path=".", label_numbers=[0, 1, 2],
    train_csv_name="train.csv", valid_csv_name="valid.csv", test_csv_name="test.csv", 
    skiprows=1, encoding="euc-kr", test_size=0.25, valid_size=0.25, random_seed=1):

    # Read raw data
    df_raw = pd.read_csv(raw_data_path, skiprows=skiprows, encoding=encoding)

    # 빈 텍스트 행 제거
    df_raw.drop( df_raw[df_raw.text.str.len() < 5].index, inplace=True)

    df_raw = df_raw.dropna()

    # Trim text and titletext to first_n_words
    df_raw['text'] = df_raw['text'].apply(trim_string)

    df_split_train = pd.DataFrame()
    df_split_valid = pd.DataFrame()
    df_split_test = pd.DataFrame()

    for ln in label_numbers:
        # Split according to label
        df_label = df_raw[df_raw['label'] == ln]

        # Train-test split
        df_full_train, df_test = train_test_split(df_label, test_size = test_size, random_state = random_seed)
        # Train-valid split
        df_train, df_valid = train_test_split(df_full_train, test_size = valid_size, random_state = random_seed)

        # Concatenate splits of different labels
        df_split_train = pd.concat([df_split_train, df_train], ignore_index=True, sort=False)
        df_split_valid = pd.concat([df_split_valid, df_valid], ignore_index=True, sort=False)
        df_split_test = pd.concat([df_split_test, df_test], ignore_index=True, sort=False)

    # Write preprocessed data
    df_split_train.to_csv(dest_path + "/" + train_csv_name, index=False, encoding="utf-8")
    df_split_valid.to_csv(dest_path + "/" + valid_csv_name, index=False, encoding="utf-8")
    df_split_test.to_csv(dest_path + "/" + test_csv_name, index=False, encoding="utf-8")


def trim_string(x, first_n_words=200):
    x = x.split(maxsplit=first_n_words)
    x = ' '.join(x[:first_n_words])
    return x


# Fields
def get_fields(tokenize=str.split):
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
    text_field = Field(tokenize=tokenize, lower=True, include_lengths=True, batch_first=True)
    fields = [('text', text_field), ('label', label_field)]

    return label_field, text_field, fields


def get_text_field(tokenize=str.split):
    text_field = Field(tokenize=tokenize, lower=True, include_lengths=True, batch_first=True)

    return text_field


def get_datasets(fields, source_path=".", train_csv="train.csv", valid_csv="valid.csv", test_csv="test.csv"):
    # TabularDataset
    train, valid, test = TabularDataset.splits(path=source_path, train=train_csv, validation=valid_csv, test=test_csv,
                                            format='CSV', fields=fields, skip_header=True)
    
    return train, valid, test


# Iterators
def get_iterators(train_data, valid_data, test_data, device, train_batch_size=5, valid_batch_size=5, test_batch_size=5):
    train_iter = BucketIterator(train_data, batch_size=train_batch_size, sort_key=lambda x: len(x.text),
                            device=device, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid_data, batch_size=valid_batch_size, sort_key=lambda x: len(x.text),
                            device=device, sort=True, sort_within_batch=True)
    test_iter = BucketIterator(test_data, batch_size=test_batch_size, sort_key=lambda x: len(x.text),
                            device=device, sort=True, sort_within_batch=True)
    
    return train_iter, valid_iter, test_iter


# Vocabulary
def get_vocablulary(text_field, train_data, min_freq=2):
    text_field.build_vocab(train_data, min_freq=min_freq)
    return text_field


def get_reverse_vocablulary_and_iter(source_path, tokenize, device, batch_size, word_min_freq):
    rev_field = ReversibleField(tokenize=tokenize, lower=True, include_lengths=True, batch_first=True)
    rev_fields = [('text', rev_field)]

    rev_pred_data = TabularDataset(path=source_path, format='CSV', fields=rev_fields, skip_header=True)

    rev_pred_iter = BucketIterator(rev_pred_data, batch_size=batch_size, sort_key=lambda x: len(x.text),
            device=device, sort=True, sort_within_batch=True)

    rev_field.build_vocab(rev_pred_data, min_freq=word_min_freq)

    return rev_field, rev_pred_iter


# Save and Load Functions
def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        print("Required Save Path")
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                    'embedding': model.embedding,
                    'vocab': model.vocab,
                    'vocab_size': model.vocab_size,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def predict_sentence(model, vocab, sentence, tokenize, device, cpu_device="cpu"):
    tokenized = [w for w in tokenize(sentence)]
    indexed = [vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1).T
    length_tensor = torch.LongTensor(length).to(cpu_device)
    prediction = model(tensor, length_tensor)
    return prediction


def load_checkpoint(load_path, model, device, optimizer=None, strict=True):

    if load_path==None:
        print("Required Load Path")
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'], strict=strict)

    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def load_pretrained_weights(load_path, device):

    if load_path==None:
        print("Required Load Path")
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    vocab = state_dict["vocab"]
    weight = state_dict["embedding"].weight
    
    return vocab, weight


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        print("Required Save Path")
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):

    if load_path==None:
        print("Required Load Path")
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']