import os
import argparse
import pandas as pd
import numpy as np
from tarfile import ENCODING

import torch

from torch.utils.data import Dataset, DataLoader
from torchtext.legacy.data import TabularDataset, BucketIterator

from model import LSTM, CNN1d
from utils import colorstr
from general import get_reverse_vocablulary_and_iter, get_text_field, get_vocablulary, sentence_prediction,\
    save_checkpoint, save_metrics, load_checkpoint, load_pretrained_weights, load_metrics

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
    parser.add_argument('--threshold', type=float, default=0.7, help='threshold')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--word-min-freq', type=int, default=1, help='voca word min frequency')
    parser.add_argument('--emb-dim', type=int, default=300, help='embedding size')
    parser.add_argument('--out-channel', type=int, default=100, help='out channel size')
    parser.add_argument('--shuffle-data', action='store_true', help='shuffle iterator')
    parser.add_argument('--reverse-field', action='store_true', help='use reverse field')

    opt = parser.parse_args()
    return opt


class PredictionDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]

        return x


def prediction(model, pred_iter, classes, device, cpu_device, rev_field, tokenize, threshold=0.5):
    model.eval()

    batch_size = pred_iter.batch_size
    iter_length = len(pred_iter) * batch_size
    cnt = batch_size
    origin_text = ""

    with torch.no_grad():
        sigmoid = torch.nn.Sigmoid()
        pred_str = None

        # get text
        if rev_field is not None:
            for (text, text_len), _ in pred_iter:
                print(f"[{cnt} / {iter_length}]", "--" * 25)
                for tt in text:
                    for t in range(len(tt)):
                        origin_text += rev_field.reverse(tt[t].unsqueeze(0).unsqueeze(0))[0]
                    output = sentence_prediction(model, model.vocab, origin_text, tokenize, device, cpu_device)
                    output= sigmoid(output)
                    output = (output > threshold).int().to(cpu_device)
                    top_idx = torch.topk(output, 1)
                    top_idx = top_idx.indices.numpy().reshape(-1)

                    pred_str = classes[top_idx[0]] if len(output[output == 1]) == 1 else "unknown"
                    print("Sentence:", colorstr(origin_text), "Prediction:", colorstr("bright_green", "bold", f"Class [{pred_str}]"))
                    origin_text = ""
                cnt += batch_size
                q = input("Next? [y/n]: ")
                if iter_length < cnt or (q != "y" and q != "Y" and q != ""):
                    print(colorstr("red", "bold", "Exit Program"))
                    exit()
        else:
            for texts in pred_iter:
                print(f"[{cnt} / {iter_length}]", "--" * 25)
                for text in texts:
                    output = sentence_prediction(model, model.vocab, text, tokenize, device, cpu_device)
                    output= sigmoid(output)
                    output = (output > threshold).int().to(cpu_device)
                    top_idx = torch.topk(output, 1)
                    top_idx = top_idx.indices.numpy().reshape(-1)

                    pred_str = classes[top_idx[0]] if len(output[output == 1]) == 1 else "unknown"
                    print("Sentence:", colorstr(text), "Prediction:", colorstr("bright_green", "bold", f"Class [{pred_str}]"))
                cnt += batch_size
                q = input("Next? [y/n]: ")
                if iter_length < cnt or (q != "y" and q != "Y" and q != ""):
                    print(colorstr("red", "bold", "Exit Program"))
                    exit()


def main(opt):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cpu_device = "cpu"

    #classes = ["hello", "sorry", "thank", "emergency", "weather"]
    classes = ["hello", "sorry", "thank"]

    print(colorstr("red", "bold", "Prediction: ") + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    m = Mecab()

    rev_field = None

    if opt.reverse_field:
        # get reverse vocab
        text_field = get_text_field(m.morphs)
        fields = [('text', text_field)]

        pred_data = TabularDataset(path=opt.source_path + opt.source_name, format='CSV', fields=fields, skip_header=True)

        pred_iter = BucketIterator(pred_data, batch_size=opt.batch_size, sort_key=lambda x: len(x.text),
                device=device, sort=False, sort_within_batch=False, shuffle=opt.shuffle_data)

        text_field = get_vocablulary(text_field, pred_data, min_freq=opt.word_min_freq)

        rev_field, rev_pred_iter = get_reverse_vocablulary_and_iter(opt.source_path + opt.source_name, m.morphs,
                                    device, opt.batch_size, opt.word_min_freq)
    else:
        # load data
        df_data = pd.read_csv(opt.source_path + opt.source_name, skiprows=0, encoding="utf-8")
        text_data = df_data["text"].to_numpy()
        pred_dataset = PredictionDataset(text_data)
        pred_iter = DataLoader(pred_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle_data)

    # load vocab
    vocab, weight = load_pretrained_weights(opt.weights_save_path + "/" + opt.best_weight_save_name, device)

    # use model list
    model_list = {
        "LSTM": LSTM(vocab, len(vocab), class_num=len(classes), embed_dim=opt.emb_dim).to(device),
        "CNN1d": CNN1d(vocab, len(vocab), class_num=len(classes), embed_dim=opt.emb_dim, n_filters=opt.out_channel).to(device)
    }

    model = model_list[opt.model_name]
    load_checkpoint(opt.weights_save_path + "/" + opt.best_weight_save_name, model, device, optimizer=None, strict=False)

    prediction(model, pred_iter, classes, device, cpu_device, rev_field, tokenize=m.morphs, threshold=opt.threshold)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)