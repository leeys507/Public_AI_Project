import os
import argparse
from tarfile import ENCODING

import torch
import torch.optim as optim

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from torchtext.legacy.data.field import ReversibleField

from model import LSTM, CNN1d
from utils import colorstr
from general import create_split_csv, get_fields, get_datasets, get_iterators, \
    get_reverse_vocablulary_and_iter, get_text_field, get_vocablulary, predict_sentence,\
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
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--word-min-freq', type=int, default=1, help='voca word min frequency')
    parser.add_argument('--emb-dim', type=int, default=300, help='embedding size')
    parser.add_argument('--out-channel', type=int, default=100, help='out channel size')
    parser.add_argument('--shuffle-data', action='store_true', help='shuffle iterator')

    opt = parser.parse_args()
    return opt


def prediction(model, pred_iter, classes, device, cpu_device, rev_field, tokenize, threshold=0.5):
    model.eval()

    batch_size = pred_iter.batch_size
    iter_length = len(pred_iter) * batch_size
    cnt = batch_size
    origin_text = ""

    with torch.no_grad():                    
        # get text
        for (text, text_len), _ in pred_iter:
            print(f"[{cnt} / {iter_length}]", "--" * 25)
            for tt in text:
                for t in range(len(tt)):
                    origin_text += rev_field.reverse(tt[t].unsqueeze(0).unsqueeze(0))[0]
                output = predict_sentence(model, model.vocab, origin_text, tokenize, device, cpu_device)
                output = (output > threshold).int().to(cpu_device)
                top_idx = torch.topk(output, 1)
                top_idx = top_idx.indices.numpy().reshape(-1)
                print("Sentence:", colorstr(origin_text), "Prediction:", colorstr("bright_green", "bold", f"Class [{classes[top_idx[0]]}]"))
                origin_text = ""
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
    text_field = get_text_field(m.morphs)
    fields = [('text', text_field)]

    vocab, weight = load_pretrained_weights(opt.weights_save_path + "/" + opt.best_weight_save_name, device)

    pred_data = TabularDataset(path=opt.source_path + opt.source_name, format='CSV', fields=fields, skip_header=True)

    pred_iter = BucketIterator(pred_data, batch_size=opt.batch_size, sort_key=lambda x: len(x.text),
            device=device, sort=False, sort_within_batch=False, shuffle=opt.shuffle_data)

    text_field = get_vocablulary(text_field, pred_data, min_freq=opt.word_min_freq)

    rev_field, rev_pred_iter = get_reverse_vocablulary_and_iter(opt.source_path + opt.source_name, m.morphs,
                                device, opt.batch_size, opt.word_min_freq)

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