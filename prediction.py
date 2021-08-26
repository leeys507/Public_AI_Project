import os
import argparse
from tarfile import ENCODING

import torch
import torch.optim as optim

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from torchtext.legacy.data.field import ReversibleField

from model import LSTM, CNN1d
from utils import colorstr
from general import create_split_csv, get_fields, get_datasets, get_iterators, get_reverse_vocablulary_and_iter, get_text_field, get_vocablulary,\
    save_checkpoint, save_metrics, load_checkpoint, load_metrics

from eunjeon import Mecab

def parse_opt():
    default_path = os.path.join(os.path.expanduser('~'), 'Desktop/') # Desktop
    weights_path = "weights/speech_text"
    save_pt = "model.pt"

    source_path = default_path + "ai_data/speech_text/csv_data/"
    source_name = "pred_data.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-save-path', type=str, default=default_path + weights_path, help='save model.pt path(s)')
    parser.add_argument('--best-weight-save-name', type=str, default=save_pt, help='save best weight name')
    parser.add_argument('--source-path', type=str, default=source_path, help='data source path')
    parser.add_argument('--source-name', type=str, default=source_name, help='source name')
    parser.add_argument('--model-name', type=str, default="LSTM", help='select model')
    parser.add_argument('--random-seed', type=int, default=1, help='random seed')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--word-min-freq', type=int, default=1, help='voca word min frequency')
    parser.add_argument('--emb-dim', type=int, default=300, help='embedding size')
    parser.add_argument('--out-channel', type=int, default=100, help='out channel size')

    opt = parser.parse_args()
    return opt


def prediction(model, pred_iter, device, cpu_device, rev_field, rev_pred_iter, threshold=0.5):
    with torch.no_grad():                    
        # get text
        for (text, text_len), _ in pred_iter:
            for tt in text:
                for t in range(len(tt)):
                    print(rev_field.reverse(tt[t].unsqueeze(0).unsqueeze(0)))
            text = text.to(device)
            text_len = text_len.to(cpu_device)
            output = model(text, text_len)
            print(output)
            output = (output > threshold).int()
            print(output)
            print("--" * 20)


def main(opt):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cpu_device = "cpu"

    #classes = ["hello", "sorry", "thank", "emergency", "weather"]
    classes = ["hello", "sorry", "thank"]

    print(colorstr("red", "bold", "Prediction: ") + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    m = Mecab()
    text_field = get_text_field(m.morphs)
    fields = [('text', text_field)]

    pred_data = TabularDataset(path=opt.source_path + opt.source_name, format='CSV', fields=fields, skip_header=True)

    pred_iter = BucketIterator(pred_data, batch_size=opt.batch_size, sort_key=lambda x: len(x.text),
            device=device, sort=True, sort_within_batch=True)

    text_field = get_vocablulary(text_field, pred_data, opt.word_min_freq)

    rev_field, rev_pred_iter = get_reverse_vocablulary_and_iter(opt.source_path + opt.source_name, m.morphs, 
                                device, opt.batch_size, opt.word_min_freq)
    
    # for (texts, text_len), _ in rev_pred_iter:
    #     for text in texts:
    #         for t in range(len(text)):
    #             print(rev_field.reverse(text[t].unsqueeze(0).unsqueeze(0)))
    #         print("--" * 20)
    # exit()

    # use model list
    model_list = {
        "LSTM": LSTM(len(text_field.vocab), class_num=len(classes), embed_dim=opt.emb_dim).to(device),
        "CNN1d": CNN1d(len(text_field.vocab), class_num=len(classes), embed_dim=opt.emb_dim, n_filters=opt.out_channel).to(device)
    }

    model = model_list[opt.model_name]
    optimizer = optim.Adam(model.parameters())

    load_checkpoint(opt.weights_save_path + "/" + opt.best_weight_save_name, model, optimizer, device)
    prediction(model, pred_iter, device, cpu_device, rev_field, rev_pred_iter, threshold=opt.threshold)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)