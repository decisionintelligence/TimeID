import torch
import torch.nn as nn
import torch.optim as optim
from models.dbc import DBC
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import seaborn as sns

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

torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logpath = os.path.join(args.cv_dir, '{}_sl{}_pl{}'.format(args.data_name, args.seq_len, args.pred_len))
    os.makedirs(logpath, exist_ok=True)
    #writer = SummaryWriter(log_dir = logpath, flush_secs = 30)
    
    seed=args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainset, trainloader = get_data(args, flag='train')
    testset, testloader = get_data(args, flag='test')

    criterion = nn.MSELoss().to(args.device)

    model = DBC(args)
    model = model.to(args.device)

    model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params':model_params}]
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    train = train_normal


    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    '''
    for epoch in tqdm(range(start_epoch, args.max_epochs + 1), desc = 'Current epoch'):
        train_loss = train(args, epoch, model, trainloader, optimizer)
        #writer.add_scalar('train_loss', train_loss, epoch)
        if epoch % args.eval_val_every == 0:
            with torch.no_grad():
                vali_loss = vali(epoch, model, testloader, args, criterion)
                #writer.add_scalar('vali_loss', vali_loss, epoch)
            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(
                epoch + 1, train_loss, vali_loss))
            filename='best_model'
            early_stopping(vali_loss, model, logpath, epoch, filename)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    '''
    epoch=0

    best_model_path = logpath + '/' + 'ckpt_best_model.t7'
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['net'])
    print("------------------------------------")
    mse, mae = test(epoch, model, testloader, args)
    print("mse: {}\tmae: {}\t".format(mse,mae))
    #writer.close()
    resinvs, trendinvs = collect_invariants(model, testloader, args)
    visualize_tsne_invariants(resinvs, trendinvs, logpath)

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
        X_2d[:len(idx), 0], X_2d[:len(idx), 1],
        c='C0', alpha=0.6, label='Seasonal'
    )
    plt.scatter(
        X_2d[len(idx):, 0], X_2d[len(idx):, 1],
        c='C1', alpha=0.6, label='Trend'
    )
    plt.legend(fontsize=12)
    plt.title('t-SNE of Seasonal and Trend Features')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(logpath, 'tsne_invariants.pdf'), dpi=300)
    plt.close()
    print(f"t-SNE 图已保存到 {logpath}")


def train_normal(args, epoch, model, trainloader, optimizer):
    model.train()

    train_loss = 0.0

    for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(trainloader):
        batch_x = batch_x.float().to(args.device)
        batch_y = batch_y.float().to(args.device)

        loss, _ = model(args, batch_x, batch_y, epoch)
 
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
    parser.add_argument('--data_name', type=str, default='weather')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of layers checkpoints')
    parser.add_argument('--percent', type=int, default=100)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')

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
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
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
    
    args = parser.parse_args()
    main(args)