
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import random

import logging
import sys
from datetime import datetime

def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)


    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip() != "":
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


logger = setup_logger()
sys.stdout = StreamToLogger(logger, logging.INFO)




#np.random.seed(0)


def set_seed(seed):
    torch.manual_seed(seed) #设置 PyTorch 的随机种子
    np.random.seed(seed) #设置 NumPy 的随机种子
    random.seed(seed)  #设置 Python 内置 random 模块的随机种子
    torch.cuda.manual_seed_all(seed)  # if using CUDA，设置其随机种子
    torch.backends.cudnn.deterministic = True #将 CuDNN（CUDA 的深度学习加速库）设置为确定性模式
    torch.backends.cudnn.benchmark = False #禁用 CuDNN 的 benchmark 模式

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

def plot_loss_data(losses, save_dir='loss_process_figures',trial_number=None):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    if trial_number is not None:
        filename = f"loss_curve_trial_{trial_number}.png"
    else:
        filename = f"loss_curve_{int(time.time())}.png"

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]

        if len(sequence.shape) == 1:
            sequence = sequence[:, np.newaxis]
        if len(label.shape) == 1:
            label = label[:, np.newaxis]

        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def calculate_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae



def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>dalaloader<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(config.data_path)
    pre_len = config.pre_len
    train_window = config.window_size


    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[1:]
    df_data = df[cols_data]


    true_data = df_data.values

    scaler = StandardScaler()
    scaler.fit(true_data)


    train_data = true_data[:int(0.7 * len(true_data))] #0% ~ 70%
    valid_data = true_data[int(0.7 * len(true_data)):int(0.85 * len(true_data))] #70% ~ 85%
    test_data = true_data[int(0.85 * len(true_data)):] #85% ~ 100%
    print("训练集尺寸:", len(train_data), "测试集尺寸:", len(test_data), "验证集尺寸:", len(valid_data))

    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    valid_data_normalized = scaler.transform(valid_data)

    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)

    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)

    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True) #worker_init_fn=lambda id: np.random.seed(42 + id)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    config.input_size = true_data.shape[1]    # 自动设置 input_size，确保与真实特征数一致

    print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
    print("通过滑动窗口共有测试集数据：", len(test_inout_seq), "转化为批次数据:", len(test_loader))
    print("通过滑动窗口共有验证集数据：", len(valid_inout_seq), "转化为批次数据:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, test_loader, valid_loader, scaler


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, nheads, dropout):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nheads, dropout=dropout)


    def forward(self, x):
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attn(x, x, x)
        return attn_output.permute(1, 0, 2)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout, nheads):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.attention = SelfAttention(n_outputs, nheads, dropout)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)  # out: [batch_size, n_outputs, seq_len]
        out = out.permute(0, 2, 1)  # out: [batch_size, seq_len, n_outputs]
        out = self.attention(out)  # out: [batch_size, seq_len, n_outputs]
        out = out.permute(0, 2, 1)  # 🔁 恢复回 [batch_size, n_outputs, seq_len] for residual
        res = x if self.downsample is None else self.downsample(x)  # res: [batch_size, n_outputs, seq_len]
        return self.relu(out + res)  #  [batch_size, seq_len, n_outputs]

"""TCN定义的核心部分"""
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, pre_len, num_channels, kernel_size, dropout, nheads):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.pre_len = pre_len

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout,nheads=nheads)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        return x[:, -self.pre_len:, :]


def train(model, args, scaler, device,train_loader):

    def nse_loss(pred, target):
        numerator = torch.sum((target - pred) ** 2)
        denominator = torch.sum((target - torch.mean(target)) ** 2)
        return numerator / (denominator + 1e-6)

    start_time = time.time()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    model.train()
    results_loss = []

    best_loss = float('inf')
    best_epoch = -1

    for epoch in tqdm(range(epochs)):
        epoch_losses = []
        for seq, labels in train_loader:
            optimizer.zero_grad()
            seq = seq.to(device)
            labels = labels.to(device)
            y_pred = model(seq)
            loss = nse_loss(y_pred, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        results_loss.append(avg_loss)
        tqdm.write(f"[INFO] Epoch {epoch + 1} / {epochs}, Train NSE Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            save_path = f"save_model_trial_{args.trial_num}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"新最佳模型已保存 @ Epoch {epoch + 1}, NSE Loss: {best_loss:.6f}")

        time.sleep(0.1)

    print(f">>>>>>>>>>>>>>>>>>>>>>模型训练完成，用时: {(time.time() - start_time) / 60:.2f} min<<<<<<<<<<<<<<<<<<")
    print(f"最佳模型出现在 Epoch {best_epoch + 1}, NSE Loss: {best_loss:.6f}")
    plot_loss_data(results_loss)  #loss results Plot

def valid(model, args, valid_loader, scaler,trial_num=None):
    tcn_model = model
    model_path = f"save_model_trial_{args.trial_num}.pth"
    tcn_model.load_state_dict(torch.load(model_path))
    tcn_model.eval()
    losss = []
    all_preds = []
    all_labels = []

    for seq, labels in valid_loader:
        if seq is None or labels is None:
            print("seq或labels为空")
            continue
        pred = tcn_model(seq)

        if pred is None:
            print("pred为空")
            continue
        if pred.shape != labels.shape:
            print("pred和labels的形状不一致")
            continue
        vlabel = scaler.inverse_transform(labels.detach().cpu().numpy())
        vpred = scaler.inverse_transform(pred.detach().cpu().numpy())

        mae = calculate_mae(vlabel, vpred)
        losss.append(mae)

        mse = np.mean((vlabel - vpred) ** 2)
        rmse = np.sqrt(mse)

        # NSE
        ss_tot = np.sum((vlabel - np.mean(vlabel)) ** 2)
        ss_res = np.sum((vlabel - vpred) ** 2)
        nse = 1 - (ss_res / ss_tot)

        all_labels.extend(vlabel.flatten())
        all_preds.extend(vpred.flatten())

    avg_mae = np.mean(np.abs(np.array(all_labels) - np.array(all_preds)))

    avg_mse = np.mean((np.array(all_labels) - np.array(all_preds)) ** 2)
    avg_rmse = np.sqrt(avg_mse)

    ss_tot = np.sum((np.array(all_labels) - np.mean(all_labels)) ** 2)
    ss_res = np.sum((np.array(all_labels) - np.array(all_preds)) ** 2)
    avg_nse = 1 - (ss_res / ss_tot)

    print("平均验证集误差MAE:", avg_mae)
    print("平均验证集误差RMSE:", avg_rmse)
    print("平均验证集误差NSE:", avg_nse)

    if trial_num is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(all_labels, label="True")
        plt.plot(all_preds, label="Prediction")
        plt.title(f"Trial {trial_num}: RMSE={avg_rmse:.2f}, NSE={avg_nse:.2f}")
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.legend()
        save_dir = "validation"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"trial_{trial_num}.png"), dpi=300)
        plt.close()

    df = pd.DataFrame({
        '真实值': all_labels,
        '预测值': all_preds,
        'MAE': [avg_mae] * len(all_labels),
        'RMSE': [avg_rmse] * len(all_labels),
        'NSE': [avg_nse] * len(all_labels)
    })

    save_path = r"I:\personal\TSDeepAR\TCNmodel\TCN3"
    trial_id = f"_trial_{trial_num}" if trial_num is not None else ""
    save_dir = os.path.join("trialresult", f"validation{trial_id}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"validation{trial_id}.xlsx")
    df.to_excel(file_path, index=False)
    print(f"验证结果已保存到 {file_path}")

    return avg_rmse, avg_nse


def tst(model, args, test_loader, scaler, trial_num=None):
    losss = []
    tcn_model = model
    model_path = f"save_model_trial_{args.trial_num}.pth"
    tcn_model.load_state_dict(torch.load(model_path))
    model.eval()
    results = []
    labels = []
    for seq, label in test_loader:
        pred = model(seq)
        pred = pred[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        mae = calculate_mae(label, pred)
        losss.append(mae)
        print("一个批次平均绝对误差MAE（真实空间）:", losss)


        mse = np.mean((label - pred) ** 2)
        rmse = np.sqrt(mse)
        ss_tot = np.sum((label - np.mean(label)) ** 2)
        ss_res = np.sum((label - pred) ** 2)
        nse = 1 - (ss_res / ss_tot)

        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])


    overall_mae = np.mean(losss)
    overall_mse = np.mean((np.array(labels) - np.array(results)) ** 2)
    overall_rmse = np.sqrt(overall_mse)
    ss_tot = np.sum((np.array(labels) - np.mean(labels)) ** 2)
    ss_res = np.sum((np.array(labels) - np.array(results)) ** 2)
    overall_nse = 1 - (ss_res / ss_tot)
    print("整个测试集平均绝对误差MAE:", overall_mae)
    print("整个测试集平均均方根误差RMSE:", overall_rmse)
    print("整个测试集平均NSE:", overall_nse)

    results = np.array(results)
    labels = np.array(labels)
    errors = results - labels
    sigma_est = np.std(errors)
    lower_bound = results - 0.674 *sigma_est
    upper_bound = results + 0.674 *sigma_est

    plt.plot(labels, label='TrueValue')
    plt.plot(results, label='Prediction')
    plt.fill_between(range(len(results)), lower_bound, upper_bound, color='gray', alpha=0.3,
                     label='50% Confidence Interval')

    plt.title("test state\nMAE: {:.2f}  RMSE: {:.2f}  NSE: {:.2f}".format(overall_mae, overall_rmse, overall_nse),
              loc='left', pad=20)
    plt.legend()

    if trial_num is not None:
        os.makedirs('test', exist_ok=True)
        plt.savefig(f'test/test_trial_{trial_num}.png', dpi=500)
        plt.close()

    df = pd.DataFrame({
        '真实值': labels,
        '预测值': results,
        'MAE': [overall_mae] * len(labels),
        'RMSE': [overall_rmse] * len(labels),
        'NSE': [overall_nse] * len(labels)
    })

    save_path =  r"I:\TSDeepAR\TCNmodel"
    os.makedirs(save_path, exist_ok=True)
    trial_id = f"_trial_{trial_num}" if trial_num is not None else ""
    save_dir = os.path.join("trialresult", f"test{trial_id}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"test{trial_id}.xlsx")
    df.to_excel(file_path, index=False)
    print(f"测试结果已保存到 {file_path}")

    return overall_rmse, overall_nse



def inspect_model_fit(model, args, train_loader, scaler):
    model = model
    trial_id = getattr(args, 'trial_num', 'manual')
    model_path = f"save_model_trial_{trial_id}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    results = []
    labels = []

    for seq, label in train_loader:
        pred = model(seq)[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])

    plt.plot(labels, label='History')
    plt.plot(results, label='Prediction')
    plt.title("inspect model fit state")
    plt.legend()
    plt.show()


    df = pd.DataFrame({
        '真实值': labels,
        '预测值': results,
    })

    save_path = r"I:TSDeepAR\TCNmodel\salt"
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, '3train_fit.xlsx')
    df.to_excel(file_path, index=False)
    print(f"测试结果已保存到 {file_path}")

def predict(model, args, device, scaler):
    df = pd.read_csv(args.data_path)
    df = df.iloc[:, 1:][-args.window_size:].values
    pre_data = scaler.transform(df)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)
    model = model
    trial_id = getattr(args, 'trial_num', 'manual')
    model_path = f"save_model_trial_{trial_id}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pred = model(tensor_pred)[0]
    pred = scaler.inverse_transform(pred.detach().cpu().numpy())

    pred_values = pred[:, -1]
    true_values = df[:, -1]
    if len(pred_values) > len(true_values):
        pred_values = pred_values[:len(true_values)]
    elif len(true_values) > len(pred_values):
        true_values = true_values[:len(pred_values)]
    errors = pred_values - true_values
    sigma_est = np.std(errors)
    mu = np.mean(pred_values)
    sigma = sigma_est


    history_length = len(df[:, -1])
    history_x = range(history_length)
    prediction_x = range(history_length - 1, history_length + len(pred[:, -1]) - 1)
    plt.plot(history_x, df[:, -1], label='History')
    plt.plot(prediction_x, pred[:, -1], marker='o', label='Prediction')


    for i in range(len(pred_values) - 1):
        conf_interval_lower = pred_values - 0.674 * sigma
        conf_interval_upper = pred_values + 0.674 * sigma
        plt.fill_between(prediction_x,conf_interval_lower,conf_interval_upper, color="orange", alpha=0.1)

    plt.axvline(history_length - 1, color='red')
    plt.title("History and Prediction")
    plt.legend()
    plt.show()

    history_data = df[:, -1]
    prediction_data = pred[:, -1]
    max_length = max(len(history_data), len(prediction_data))
    history_data = np.pad(history_data, (0, max_length - len(history_data)), constant_values=np.nan)
    prediction_data = np.pad(prediction_data, (0, max_length - len(prediction_data)), constant_values=np.nan)

    output_df = pd.DataFrame({
        "History": history_data,
        "Prediction": prediction_data,
    })

    save_path = r"I:TSDeepAR\TCNmodel\TCN3\salt"
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, '4predict.xlsx')

    output_df.to_excel(file_path, index=False)
    print(f"测试结果已保存到 {file_path}")


if __name__ == '__main__':
    set_seed(42 + id)  #需要固定随机种子数时再打开这一行
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='TCN', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=336, help="时间窗口(时间步)大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=168, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
    parser.add_argument('-data_path', type=str, default='.\salt\inputdata\saltMA.csv', help="你的数据数据地址")
    parser.add_argument('-target', type=str, default='OT', help='你需要预测的特征列，这个值会最后保存在csv文件里')
  #  parser.add_argument('-input_size', type=int, default=1, help='你的特征个数，不算时间那一列。如果是多变量时间序列，则需要调整')
    parser.add_argument('-feature', type=str, default='S', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
    parser.add_argument('-model_dim', type=list, default=[180,80,175], help='这个地方是这个TCN卷积的关键部分,它代表了TCN的层数'
                                                                             '我这里输入list中包含三个元素那么我的TCN就是三层，这个根据你的数据复杂度来设置'
                                                                             '层数越多对应数据越复杂但是不要超过5层。这里数字代表每层的通道数')
    # learning
    parser.add_argument('-lr', type=float, default=0.000376161953185098, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.310819878613283, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=28, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=16, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')

    # model
    parser.add_argument('-hidden_size', type=int, default=128, help="隐藏层单元数")
    parser.add_argument('-kernel_sizes', type=int, default=4)
    parser.add_argument('-laryer_num', type=int, default=5)
    # device5
    parser.add_argument('-use_gpu', type=bool, default=False)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-valid', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)
    args = parser.parse_args()

    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print("使用设备:", device)

    train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)

    if args.feature in ['MS', 'S']:
        args.output_size = 1
    else:
        args.output_size = args.input_size


    args.use_optuna = True  # 启用 Optuna 优化

    if args.use_optuna:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>开始 Optuna 超参数优化<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        from optuna_optimization import optimize_hyperparameters

        best_params = optimize_hyperparameters(args, device)
        args.model_dim = [args.input_size, best_params['model_dim_2'], best_params['model_dim_3']]
        args.kernel_sizes = best_params['kernel_sizes']
        args.drop_out = best_params['drop_out']
        args.nheads = best_params['nheads']
        args.lr = best_params['lr']
        args.batch_size = best_params['batch_size']
        args.epochs = best_params['epochs']
        args.best_params = best_params
    else:
        args.nheads = getattr(args, 'nheads', 2)
        args.trial_num = 'manual'

    train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)


    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = TemporalConvNet(args.input_size, args.output_size, args.pre_len, args.model_dim, args.kernel_sizes, args.drop_out, args.nheads).to(
            device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")


    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device,train_loader)
    if args.valid:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型验证<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        valid(model, args, test_loader, scaler)
    if args.test:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        tst(model, args, test_loader, scaler)
    if args.inspect_fit:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        inspect_model_fit(model, args, train_loader, scaler)
    if args.predict:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        predict(model, args, device, scaler)
    plt.show()