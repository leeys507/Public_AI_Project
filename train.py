import time
import datetime
import os
import argparse
import numpy as np

import torch
import torch.optim as optim
# Models
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import colorstr, create_directory
from general import create_split_csv, get_fields, get_datasets, get_iterators, get_vocablulary,\
    save_checkpoint, save_metrics, load_checkpoint, load_metrics

def parse_opt():
    default_path = os.path.join(os.path.expanduser('~'), 'Desktop/') # Desktop
    weights_path = "weights/speech_text"
    saved_pt = "model.pt"

    source_path = default_path + "ai_data/speech_text/csv_data/"
    source_name = "data.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-save-path', type=str, default=default_path + weights_path, help='save model.pt path(s)')
    parser.add_argument('--best-weight-save-name', type=str, default=saved_pt, help='save best weight name')
    parser.add_argument('--source-path', type=str, default=source_path, help='data source path')
    parser.add_argument('--source-name', type=str, default=source_name, help='source name')
    parser.add_argument('--outputs-path', type=str, default=source_path, help='spkit data outputs path name')
    parser.add_argument('--train-data-save-name', type=str, default="train.csv", help='save train data name')
    parser.add_argument('--valid-data-save-name', type=str, default="valid.csv", help='save valid data name')
    parser.add_argument('--test-data-save-name', type=str, default="test.csv", help='save test data name')
    parser.add_argument('--test-size', type=float, default=0.1, help='split test size')
    parser.add_argument('--valid-size', type=float, default=0.1, help='split valid size')
    parser.add_argument('--random-seed', type=int, default=1, help='random seed')
    parser.add_argument('--train-batch-size', type=int, default=5, help='train loader batch size')
    parser.add_argument('--valid-batch-size', type=int, default=5, help='valid loader batch size')
    parser.add_argument('--test-batch-size', type=int, default=5, help='test loader batch size')
    parser.add_argument('--word-min-freq', type=int, default=1, help='voca word min frequency')
    parser.add_argument('--epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--eval-every', type=int, default=3, help='train every evaluation')
    parser.add_argument('--test', action='store_true', help='test after training')
    parser.add_argument('--test-only', action='store_true', help='test only')
    parser.add_argument('--test-threshold', type=float, default=0.5, help='test threshold')

    opt = parser.parse_args()
    return opt


class LSTM(nn.Module):

    def __init__(self, text_field, class_num=3, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
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
        #text_out = torch.sigmoid(text_fea)

        return text_out


def start_train(model,
        optimizer,
        train_loader, #train_iter
        valid_loader, # valid_iter
        device,
        cpu_device,
        criterion = nn.CrossEntropyLoss(),
        num_epochs = 5,
        eval_every = 1, # len(train_iter) // 2
        weights_save_path = ".",
        save_best_model_name = "model.pt",
        best_valid_loss = float("Inf"),
    ):

    log_path = weights_save_path + "/log/"
    loss_log_name = "loss_log.txt"
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    best_accuracy = 0.0

    create_directory(log_path)
    log_file = open(log_path + loss_log_name, "w")
    log_file.write("Loss in Last Validation of Every Epoch\n")

    # training loop
    model.train()

    t0 = time.time()
    for epoch in range(num_epochs):
        for ((text, text_len), labels), _ in train_loader:           
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(cpu_device)
            output = model(text, text_len)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()

                total_acc, total_count = 0, 0
                with torch.no_grad():                    
                    # validation loop
                    for ((text, text_len), labels), _ in valid_loader:
                        labels = labels.to(device)
                        text = text.to(device)
                        text_len = text_len.to(cpu_device)
                        output = model(text, text_len)

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                        total_acc += (output.argmax(1) == labels).sum().item()
                        total_count += labels.size(0)

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(weights_save_path + "/" + "min_loss.pt", model, optimizer, best_valid_loss)
                    save_metrics(weights_save_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

                if best_accuracy < (total_acc/total_count):
                    best_accuracy = total_acc/total_count
                    saving_best_model_path = weights_save_path + "/" + save_best_model_name

                    print("--" * 25)
                    print(f'Valid Accuracy: {best_accuracy:.4f}')
                    print(f"Saving Model(Path): {saving_best_model_path}")
                    print("--" * 25)
        # one epoch end ----------------------------------------------------------------------------------------------
        log_file.write(f"Epoch {epoch} | Avg Train Loss: {average_train_loss:.4f} | Avg Valid Loss: {average_valid_loss:.4f}")

    log_file.close()
    # train loop end -------------------------------------------------------------------------------------------------
    
    save_checkpoint(weights_save_path + '/last.pt', model, optimizer, best_valid_loss)
    save_metrics(weights_save_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    total_time = time.time() - t0
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Total Traing Time:", f"{total_time_str}")
    print('Finished Training!')


    # Evaluation Function

def evaluate(model, test_loader, device, cpu_device, threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for ((text, text_len), labels), _ in test_loader:           
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(cpu_device)
            output = model(text, text_len)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
    
    print(colorstr("Classification Report:"))
    y_pred = np.argmax(y_pred, axis=1)
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], digits=4))
    
    # cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    # ax= plt.subplot()
    # sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    # ax.set_title('Confusion Matrix')

    # ax.set_xlabel('Predicted Labels')
    # ax.set_ylabel('True Labels')

    # ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    # ax.yaxis.set_ticklabels(['FAKE', 'REAL'])


def main(opt):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cpu_device = "cpu"

    #classes = ["hello", "sorry", "thank", "emergency", "weather"]
    classes = ["hello", "sorry", "thank"]

    print(colorstr("red", "bold", 'train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    # train, valid, test csv 생성
    create_split_csv(opt.source_path + opt.source_name, opt.outputs_path, 
        opt.train_data_save_name, opt.valid_data_save_name, opt.test_data_save_name, 
        test_size=opt.test_size, valid_size=opt.valid_size, random_seed=opt.random_seed)

    # data 전처리 정의, 형태소 분석
    label_field, text_field, fields = get_fields()
    
    # train, valid, test data 읽어 dataset 생성
    train_data, valid_data, test_data = get_datasets(fields=fields, source_path=opt.outputs_path, 
        train_csv=opt.train_data_save_name, valid_csv=opt.valid_data_save_name, test_csv=opt.test_data_save_name)
    
    # data loader
    train_iter, valid_iter, test_iter = get_iterators(train_data, valid_data, test_data, 
        device, opt.train_batch_size, opt.valid_batch_size, opt.test_batch_size)

    # vocablulary 생성, 단어 정수 mapping
    text_field = get_vocablulary(text_field, train_data, opt.word_min_freq)

    if opt.test_only == False:
        # model
        model = LSTM(text_field, class_num=len(classes)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

        start_train(model, optimizer, train_iter, valid_iter, device, cpu_device, 
            num_epochs=opt.epochs, eval_every=opt.eval_every, 
            weights_save_path=opt.weights_save_path, save_best_model_name=opt.best_weight_save_name)
    
    if opt.test_only or opt.test:
        best_model = LSTM(text_field, class_num=len(classes)).to(device)
        optimizer = optim.Adam(best_model.parameters(), lr=opt.lr)

        load_checkpoint(opt.weights_save_path + "/" + opt.best_weight_save_name, best_model, optimizer, device)
        evaluate(best_model, test_iter, device, cpu_device, threshold=opt.test_threshold)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)