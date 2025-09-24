import torch
import torch.nn as nn
import torch.optim as optim
from models.dbc_sfda import DBC
from models.GPT4TS import GPT4TS
from models.Transformer import Model as Transformer
import numpy as np
from lora import apply_lora_to_model

from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import tqdm
from tqdm import tqdm
import os
import argparse

from process.data_factory import get_data
from process.metrics import metric
from process.tools import EarlyStopping

import random
from torch.utils.data import Subset, DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import seaborn as sns

torch.multiprocessing.set_sharing_strategy('file_system')


def activate_specific_modules(model):
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 只激活decomp_module中的关键层
    for name, param in model.decomp_module.named_parameters():
        if 'moving_avg' in name:  # 只激活分解模块中的移动平均层
            param.requires_grad = True

    # 只激活最后一层encoder和输出层
    activate_last_layer_only(model.model_res)
    activate_last_layer_only(model.model_trend)

    # 打印统计信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params}, 总参数: {all_params}, 比例: {trainable_params / all_params:.4f}")


def activate_last_layer_only(model_part):
    # 激活最后一层encoder
    if hasattr(model_part, 'backbone') and hasattr(model_part.backbone, 'encoder'):
        if hasattr(model_part.backbone.encoder, 'layers') and len(model_part.backbone.encoder.layers) > 0:
            last_layer = model_part.backbone.encoder.layers[-1]
            for param in last_layer.parameters():
                param.requires_grad = True

    # 激活head层
    if hasattr(model_part, 'head'):
        for param in model_part.head.parameters():
            param.requires_grad = True

def visualize_tsne(preds: np.ndarray, trues: np.ndarray, logpath, sample_num=1000):
    """
    preds: (N, L, D) numpy array of model 预测
    trues: (N, L, D) numpy array of真实值
    sample_num: 从 N*L*D 中随机抽样点的数量
    """
    # 1. 平铺成 (N*L, D)
    N, L, D = preds.shape
    preds_flat = preds.reshape(-1, D)
    trues_flat = trues.reshape(-1, D)
    
    # 2. 选择部分点做可视化（避免数据量过大）
    total = N*L
    idx = np.random.choice(total, min(sample_num, total), replace=False)
    X = np.concatenate([preds_flat[idx], trues_flat[idx]], axis=0)
    
    # 3. 构造 label：0 = GroundTruth，1 = Prediction
    labels = np.array([1]*len(idx) + [0]*len(idx))
    
    # 4. t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto',
                init='random', random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # 5. 绘图
    plt.figure(figsize=(6, 6))
    plt.scatter(
        X_2d[:len(idx), 0], X_2d[:len(idx), 1],
        c='C1', alpha=0.6, label='Prediction'
    )
    plt.scatter(
        X_2d[len(idx):, 0], X_2d[len(idx):, 1],
        c='C0', alpha=0.6, label='Ground Truth'
    )
    plt.ylim(np.min(X_2d[:, 1]) - 0.2*np.max(X_2d[:, 1]) , np.max(X_2d[:, 1]) + 0.6 * np.max(X_2d[:, 1]))
    plt.legend(fontsize=20, loc='upper left')
    #plt.title('t-SNE: Pred vs True')
    #plt.xlabel('Dim 1')
    #plt.ylabel('Dim 2')

    # 去掉刻度
    plt.xticks([])
    plt.yticks([])

    # 只保留下边和左边的坐标轴边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.tick_params(width=1.5)
    plt.tight_layout()

    plt.savefig(os.path.join(logpath,'tsne.pdf'), dpi=300)
    plt.close()
    print(f"t-SNE 图已保存到 {logpath}")

def plot_pred_vs_true(preds, trues, logpath, sample_idx=0, feature_idx=0):
    """
    可视化某个样本的预测值和真实值
    Args:
        preds: numpy, (N, pred_len, D)
        trues: numpy, (N, pred_len, D)
        sample_idx: 第几个样本
        feature_idx: 第几个特征
    """
    pred = preds[sample_idx, :, feature_idx]
    true = trues[sample_idx, :, feature_idx]
    time = np.arange(len(pred))  # 时间步

    plt.figure(figsize=(7, 7))
    plt.plot(time, true, label="Ground Truth", color="blue", linewidth=4)
    plt.plot(time, pred, label="Prediction", color="red", linestyle="--", linewidth=4)
    # 调整纵坐标范围，让曲线更贴合
    plt.ylim(min(true.min(), pred.min()) - 0.2, 
            max(true.max(), pred.max()) + 0.8)
    ymin, ymax = plt.ylim()
    yticks = np.linspace(ymin, ymax, 5)  # 生成 5 个刻度
    plt.yticks(yticks)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.title(f"Sample {sample_idx}, Feature {feature_idx}")
    plt.legend(fontsize=20)
    # 坐标轴线条加粗
    plt.tick_params(width=1.5)
    #plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(logpath, 's{}_f{}.pdf'.format(sample_idx, feature_idx)), dpi=300)
    plt.close()
    print(f"Saved plot to {logpath}")


def visualize_heatmap(mat: np.ndarray, out_path: str):
    """
    mat: [L, D] 单个样本的二维矩阵
    """
    plt.figure(figsize=(7,6))
    plt.imshow(mat, aspect='auto')
    #plt.colorbar(label='Value')
    # 计算纵坐标的五等分位置
    L = mat.shape[0]  # 行数
    tick_positions = np.linspace(0, L-1, 5, dtype=int)  # 生成五个等间距的位置
    tick_labels = [f'{int(pos)}' for pos in tick_positions]  # 创建标签

    # 计算横坐标的七等分位置
    D = mat.shape[1]  # 列数
    tick_positions_x = np.linspace(0, D-1, 7, dtype=int)  # 生成七个等间距的横坐标位置
    tick_labels_x = [f'{int(pos)}' for pos in tick_positions_x]  # 创建横坐标标签

    plt.xticks(ticks=tick_positions_x, labels=tick_labels_x, fontsize=20)
    plt.yticks(ticks=tick_positions, labels=tick_labels, fontsize=20)
    plt.xlabel('Feature dim', fontsize=20)
    plt.ylabel('Time step', fontsize=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved heatmap to {out_path}")

def visualize_tsne_invariants(seasonal_features: np.ndarray, trend_features: np.ndarray, logpath: str, sample_num: int = 1000):
    """
    绘制季节和趋势特征的 t-SNE 图
    Args:
        seasonal_features: (N, L, D) numpy array of seasonal features
        trend_features: (N, L, D) numpy array of trend features
        logpath: 路径保存 t-SNE 图
        sample_num: 从 N*L*D 中随机抽样点的数量
    """
    # 1. 平铺成 (N*L, D)
    N, L, D = seasonal_features.shape
    seasonal_flat = seasonal_features.reshape(-1, D)
    trend_flat = trend_features.reshape(-1, D)
    
    # 2. 选择部分点做可视化（避免数据量过大）
    total = N * L
    idx = np.random.choice(total, min(sample_num, total), replace=False)
    X_seasonal = seasonal_flat[idx]
    X_trend = trend_flat[idx]
    
    # 3. 构造 label：0 = Seasonal，1 = Trend
    labels = np.array([0] * len(idx) + [1] * len(idx))
    X = np.concatenate([X_seasonal, X_trend], axis=0)
    
    # 4. t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random', random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # 5. 绘图
    plt.figure(figsize=(6, 6))
    plt.scatter(
        X_2d[:len(idx), 1], X_2d[:len(idx), 0],  # 交换 x 和 y
        c='C0', alpha=1, label='Seasonal'
    )
    plt.scatter(
        X_2d[:len(idx), 1], X_2d[:len(idx), 0],  # 交换 x 和 y
        c='C1', alpha=0, label='Trend'
    )
    plt.legend(fontsize=12)
    plt.title('t-SNE of Seasonal and Trend Features')
    plt.xlabel('Dimension 2')  # 更新标签
    plt.ylabel('Dimension 1')  # 更新标签
    plt.tight_layout()
    plt.savefig(os.path.join(logpath, 'tsne_invariants.pdf'), dpi=300)
    plt.close()
    print(f"t-SNE 图已保存到 {logpath}")

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = args.batch_size
    root_path = args.root_path
    data_path = args.data_path
    seq_len = args.seq_len
    pred_len = args.pred_len
    data = args.data

    logpath = os.path.join(args.cv_dir, '{}_sl{}_pl{}'.format(args.data_name, args.seq_len, args.pred_len))
    os.makedirs(logpath, exist_ok=True)
    #writer = SummaryWriter(log_dir = logpath, flush_secs = 30)
    
    seed=args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    criterion = nn.MSELoss().to(args.device)

    model_ViL = GPT4TS(args, args.device)
    #model_ViL = Transformer(args).half().to(args.device)
    for name, param in model_ViL.named_parameters():
        param.requires_grad_(False)

    model = DBC(args)
    model = model.to(args.device)

    logpath_source = os.path.join(args.cv_dir, '{}_sl{}_pl{}'.format(args.source_data_name, args.source_seq_len, args.source_pred_len))
    model_path = logpath_source + '/' + 'ckpt_best_model.t7'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['net'])



    params = model_ViL.parameters()
    model_ViL_optim = torch.optim.Adam(params, lr=args.lr)

    train = train_normal



    args.root_path = args.source_root_path
    args.data_path = args.source_data_path
    args.seq_len = args.source_seq_len
    args.pred_len = args.source_pred_len
    args.batch_size = args.source_batch_size
    args.data = args.source_data

    testset, testloader = get_data(args, flag='test')
    model.eval()
    with torch.no_grad():
        outputs_source_list = []
        for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(testloader):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float()

            _, outputs_source= model(args, batch_x, batch_y, 0)
            f_dim = -1 if args.features == 'MS' else 0
            outputs_source = outputs_source[:, -args.pred_len:, f_dim:]
            outputs_source_list.append(outputs_source.detach().clone())

    args.root_path = root_path
    args.data_path = data_path
    args.seq_len = seq_len
    args.pred_len = pred_len
    args.batch_size = batch_size
    args.data = data

    trainset, trainloader = get_data(args, flag='train')
    testset, testloader = get_data(args, flag='test')

    #activate_specific_modules(model)

    if args.use_lora:
        print(f"use LoRA, rank={args.lora_rank}, alpha={args.lora_alpha}")
        replaced_modules = apply_lora_to_model(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            trainable_orig=args.lora_trainable_orig,
            target_modules=args.lora_target_modules.split(','),
            exclude_modules=args.lora_exclude_modules.split(',')
        )
        print(f"已对 {len(replaced_modules)} 个模块应用LoRA")
        model = model.to(args.device)

    if args.use_lora:
        # 分离LoRA参数和其他参数
        lora_params = []
        other_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'lora_' in name:
                    lora_params.append(param)
                else:
                    other_params.append(param)

        # 为LoRA参数使用更大的学习率
        param_groups = [
            {'params': other_params, 'lr': args.lr},
            {'params': lora_params, 'lr': args.lr * args.lora_lr_multiplier}
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=args.wd)
    else:
        model_params = [param for name, param in model.named_parameters() if param.requires_grad]
        optim_params = [{'params': model_params}]
        optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    start_epoch = 0
    '''
    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        train_loss = train(args, epoch, model, model_ViL, trainloader, optimizer, model_ViL_optim, outputs_source_list)
        #writer.add_scalar('train_loss', train_loss, epoch)
        if epoch % args.eval_val_every == 0:
            with torch.no_grad():
                vali_loss = vali(epoch, model, testloader, args, criterion)
                #writer.add_scalar('vali_loss', vali_loss, epoch)
            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(
                epoch + 1, train_loss, vali_loss))
            filename='best_model_{}'.format(args.source_data_name)
            early_stopping(vali_loss, model, logpath, epoch, filename)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    '''
    epoch = 0

    best_model_path = logpath + '/' + 'ckpt_best_model_{}.t7'.format(args.source_data_name)
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['net'])
    print("------------------------------------")
    mse, mae = test(epoch, model, testloader, args)
    print("mse: {}\tmae: {}\t".format(mse,mae))
    #writer.close()
    
    # 额外收集 preds 和 trues
    #preds, trues = collect_preds_and_trues(model, testloader, args)
    '''
    for i in range(3):
        for j in range(3):
            plot_pred_vs_true(preds, trues, logpath, sample_idx=i, feature_idx=j)
    '''
    #visualize_tsne(preds, trues, logpath)
    
    resinvs, trendinvs = collect_invariants(model, testloader, args)
    #visualize_tsne_invariants(resinvs, trendinvs, logpath)
    '''
    # 可视化第 0 个样本
    visualize_heatmap(
        resinvs[0],
        os.path.join(logpath, 'resinvariant_heatmap_sample0.pdf')
    )
    visualize_heatmap(
        trendinvs[0],
        os.path.join(logpath, 'trendinvariant_heatmap_sample0.pdf')
    )
    '''


def collect_invariants(model, dataloader, args):
    """
    返回 numpy 数组：resinvs [N, L, D], trendinvs [N, L, D]
    """
    model.eval()
    all_r, all_t = [], []
    with torch.no_grad():
        for batch_x, _, _, _ in dataloader:
            bx = batch_x.float().to(args.device)
            r_inv, t_inv = model.extract_invariants(bx)
            all_r.append(r_inv.cpu().numpy())
            all_t.append(t_inv.cpu().numpy())
    resinvs  = np.concatenate(all_r, axis=0)
    trendinvs = np.concatenate(all_t, axis=0)
    return resinvs, trendinvs


def collect_preds_and_trues(model, testloader, args):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, _, _ in testloader:
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float()
            _, outputs = model(args, batch_x, batch_y, 0)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:].detach().cpu().numpy()
            trues = batch_y[:, -args.pred_len:, f_dim:].numpy()
            all_preds.append(outputs)
            all_trues.append(trues)
    preds = np.concatenate(all_preds, axis=0)  # (N, L, D)
    trues = np.concatenate(all_trues, axis=0)
    return preds, trues

def train_normal(args, epoch, model, model_ViL, trainloader, optimizer, model_ViL_optim, outputs_source_list):
    model.train()

    train_loss = 0.0

    for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(trainloader):
        batch_x = batch_x.float().to(args.device)
        batch_y = batch_y.float().to(args.device)

        batch_x_mark = batch_x_mark.float().to(args.device)
        batch_y_mark = batch_y_mark.float().to(args.device)
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)

        loss, _ = model(args, batch_x, batch_y, epoch, model_ViL, model_ViL_optim, outputs_source_list, idx, batch_x_mark, dec_inp, batch_y_mark)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss/len(trainloader)

    return train_loss


def vali(epoch, model, testloader, args, criterion):
    model.eval()

    total_loss = []

    for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(testloader):
        batch_x = batch_x.float().to(args.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(args.device)
        batch_y_mark = batch_y_mark.float().to(args.device)

        _, outputs= model(args, batch_x, batch_y, epoch)

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
        outputs = outputs.detach().cpu()
        batch_y = batch_y.detach().cpu()
        pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
        true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
        loss = criterion(pred, true)
        total_loss.append(loss)

    total_loss = np.average(total_loss)

    return total_loss

def test(epoch, model, testloader, args):
    preds = []
    trues = []

    model.eval()
    with torch.no_grad():
        for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(testloader):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float()

            _, outputs= model(args, batch_x, batch_y, epoch)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
            outputs = outputs.detach().cpu()
            batch_y = batch_y.detach().cpu()
            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
            preds.append(pred.numpy())
            trues.append(true.numpy())

    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

    return mse,mae


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--logpath', default=None, help='Path to dir where to logs are stored (test only)')
    parser.add_argument('--cv_dir', default='logs/', help='dir to save checkpoints and logs to')
    parser.add_argument('--load', default=None, help='path to checkpoint to load from')
    parser.add_argument('--seed', type=int, default=0, help='seed')

    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=300, help='dimension of share embedding space')
    parser.add_argument('--drop', type=float,default=0.5, help='drop rate')
    parser.add_argument('--res_epoch', type=float,default=10, help='res_epoch')
    parser.add_argument('--lambda_rep', type=float, default=1/8, help='weight of rep losses at the representation level')
    parser.add_argument('--lambda_grad', type=float, default=1/2, help='weight of grad losses at the gradient level')
    parser.add_argument('--lambda_rec', type=float, default=1/2, help='weight of rec losses at the erd')
    parser.add_argument('--lambda_res', type=float, default=1/4, help='weight of res losses at the erd')
    parser.add_argument('--IIC_PAR', type=float, default=1.3, help='weight of iic losses')

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=5e-5,help="Learning rate")
    parser.add_argument('--wd', type=float, default=5e-5,help="Weight decay")
    parser.add_argument('--save_every', type=int, default=10000,help="Frequency of snapshots in epochs")
    parser.add_argument('--eval_val_every', type=int, default=1,help="Frequency of eval in epochs")
    parser.add_argument('--max_epochs', type=int, default=800,help="Max number of epochs")

    # basic config
    #parser.add_argument('--model_id', type=str, default='test', help='layers id')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--source_data', type=str, default='custom', help='source dataset type')
    parser.add_argument('--data_name', type=str, default='weather')
    parser.add_argument('--source_data_name', type=str, default='weather')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--source_root_path', type=str, default='./dataset/', help='root path of the source data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--source_data_path', type=str, default='weather.csv', help='source data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of layers checkpoints')
    parser.add_argument('--percent', type=int, default=100)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--source_seq_len', type=int, default=96, help='source input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')
    parser.add_argument('--source_pred_len', type=int, default=336, help='source prediction sequence length')

    # DLinear
    # parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=1, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=21,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of layers')
    parser.add_argument('--d_model_FPT', type=int, default=128, help='dimension of layers')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--source_batch_size', type=int, default=128, help='batch size of source train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='cuda:0', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See process/tools for usage')

    parser.add_argument('--TTA_STEPS', type=int, default=1)

    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)

    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)

    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')

    # LoRA相关参数
    parser.add_argument('--use_lora', action='store_true', help='是否使用LoRA微调')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA的秩')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA的alpha参数')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA的dropout率')
    parser.add_argument('--lora_target_modules', type=str, default='Linear,Conv1d',
                        help='要应用LoRA的目标模块，用逗号分隔')
    parser.add_argument('--lora_trainable_orig', action='store_true', help='是否同时训练原始权重')
    parser.add_argument('--lora_exclude_modules', type=str, default='bias,ln,norm',
                        help='不应用LoRA的模块名称，用逗号分隔')
    parser.add_argument('--lora_lr_multiplier', type=float, default=5.0,
                        help='LoRA参数学习率倍数')

    
    args = parser.parse_args()
    main(args)