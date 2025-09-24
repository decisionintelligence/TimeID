import torch
import torch.fft
import torch.nn as nn
import torch.autograd as autograd
from torch import Tensor
from .layers.stdistillation_backbone import TSFE_backbone
from .layers.stdistillation_layers import series_decomp
from typing import Optional


class DBC(nn.Module):

    def __init__(self, args, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs): 
        super(DBC, self).__init__()
        configs = args
        self.args = args

        self.train_forward = self.train_forward_closed

        self.emb_dim=args.emb_dim

        self.drop = args.drop
        
        self.criterion = nn.MSELoss().to(args.device)
        self.res_epoch = args.res_epoch

        self.lambda_rep = args.lambda_rep
        self.lambda_grad = args.lambda_grad
        self.lambda_rec = args.lambda_rec
        self.lambda_res = args.lambda_res

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # layers
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size, args.dropout_rate)
            self.model_trend = TSFE_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = TSFE_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)

        else:
            self.model = TSFE_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        # self.fc = nn.Linear(target_window, configs.labels)


    def decompose_fourier(self, outputs, seasonal_ratio=0.2):
        """
        使用傅里叶变换分解季节性和趋势
        Args:
            outputs (Tensor): 输入信号，形状为 [Batch, Input length, Channel]
            seasonal_ratio (float): 季节性成分的比例，默认为 20%
        Returns:
            seasonal (Tensor): 季节性成分
            trend (Tensor): 趋势成分
        """
        
        # 傅里叶变换
        fft = torch.fft.rfft(outputs, dim=1)
        
        # 分离季节性和趋势
        n = fft.shape[1]
        seasonal_cutoff = int(n * seasonal_ratio)
        seasonal_fft = torch.zeros_like(fft)
        trend_fft = torch.zeros_like(fft)
        
        # 季节性成分（高频）
        seasonal_fft[:, seasonal_cutoff:-seasonal_cutoff] = fft[:, seasonal_cutoff:-seasonal_cutoff]
        
        # 趋势成分（低频）
        trend_fft[:, :seasonal_cutoff] = fft[:, :seasonal_cutoff]
        trend_fft[:, -seasonal_cutoff:] = fft[:, -seasonal_cutoff:]
        
        # 逆傅里叶变换
        seasonal = torch.fft.irfft(seasonal_fft, dim=1)
        trend = torch.fft.irfft(trend_fft, dim=1)
        
        return seasonal, trend

    @torch.no_grad()
    def extract_invariants(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        同时返回 res_invariant 和 trend_invariant，shape 都是 [B, L, emb_dim]
        """
        # 1. 分解
        res_init_p, trend_init_p = self.decomp_module(x)        # [B, L, C]
        res_init_pp, trend_init_pp = self.decomp_module(x)        # [B, L, C]
        # permute 到 [B, C, L]
        res_init_p, trend_init_p = res_init_p.permute(0,2,1), trend_init_p.permute(0,2,1)  # x: [Batch, Channel, Input length]
        res_init_pp, trend_init_pp = res_init_pp.permute(0,2,1), trend_init_pp.permute(0,2,1)

        # 2. 两次前向，计算 diff mask（这里只用 res 举例，trend 同理或一起做）
        # —— res branch
        res_p  = self.model_res(res_init_p).permute(0,2,1)     # [B, L, emb]
        res_pp = self.model_res(res_init_pp).permute(0,2,1)
        diff_r = torch.abs(res_p - res_pp)                  # [B, L, emb_dim]
        perct_obj = torch.sort(diff_r)[0][:, :, int(self.drop * self.emb_dim)]
        perct_obj = perct_obj.unsqueeze(-1).repeat(1, 1, self.emb_dim)
        mask_r = diff_r.lt(perct_obj).float()
        #k_r = int(self.drop * self.emb_dim)
        #thr_r, _ = torch.kthvalue(diff_r, k_r, dim=2, keepdim=True)
        #mask_r = (diff_r < thr_r).float()                   # [B, L, emb_dim]

        # —— trend branch
        trend_p  = self.model_trend(trend_init_p).permute(0,2,1)
        trend_pp = self.model_trend(trend_init_pp).permute(0,2,1)
        diff_t   = torch.abs(trend_p - trend_pp)
        perct_obj = torch.sort(diff_t)[0][:, :, int(self.drop * self.emb_dim)]
        perct_obj = perct_obj.unsqueeze(-1).repeat(1, 1, self.emb_dim)
        mask_t = diff_t.lt(perct_obj).float()
        #k_t = int(self.drop * self.emb_dim)
        #thr_t, _ = torch.kthvalue(diff_t, k_t, dim=2, keepdim=True)
        #mask_t = (diff_t < thr_t).float()
        
        # 3. 应用 mask + 再次分解
        x_r_masked = x * mask_r
        x_t_masked = x * mask_t
        res_inv, _   = self.decomp_module(x_r_masked)      # [B, L, C]
        res_inv = res_inv.permute(0,2,1)                            # [B,enc_in,L]
        res_inv = self.model_res(res_inv).permute(0,2,1)         # [B,L,emb_dim]
        _, trend_inv = self.decomp_module(x_t_masked)
        trend_inv = trend_inv.permute(0,2,1)                            # [B,enc_in,L]
        trend_inv = self.model_trend(trend_inv).permute(0,2,1)
        
        return res_inv, trend_inv

    def train_forward_closed(self, args, x, y, epoch):
        if not x.requires_grad:
            x = x.requires_grad_()

        res_init_p, trend_init_p = self.decomp_module(x)
        res_init_pp, trend_init_pp = self.decomp_module(x)

        res_init_p, trend_init_p = res_init_p.permute(0,2,1), trend_init_p.permute(0,2,1)  # x: [Batch, Channel, Input length]
        res_init_pp, trend_init_pp = res_init_pp.permute(0,2,1), trend_init_pp.permute(0,2,1)
        res_p = self.model_res(res_init_p)
        res_pp = self.model_res(res_init_pp)
        trend_p = self.model_trend(trend_init_p)
        trend_pp = self.model_trend(trend_init_pp)
        res_p, trend_p = res_p.permute(0,2,1), trend_p.permute(0,2,1)  # x: [Batch, Input length, Channel]
        res_pp, trend_pp = res_pp.permute(0,2,1), trend_pp.permute(0,2,1)

        res_g = autograd.grad((res_p * 1).sum(), x, retain_graph=True)[0]
        res_g_p = autograd.grad((res_pp * 1).sum(), x, retain_graph=True)[0]

        trend_g = autograd.grad((trend_p * 1).sum(), x, retain_graph=True)[0]
        trend_g_p = autograd.grad((trend_pp * 1).sum(), x, retain_graph=True)[0]

        diff_attr = torch.abs(res_g - res_g_p)
        perct_attr = torch.sort(diff_attr)[0][:, :, int(self.drop * self.emb_dim)]
        perct_attr = perct_attr.unsqueeze(-1).repeat(1, 1, self.emb_dim)
        mask_attr = diff_attr.lt(perct_attr.to(args.device)).float()

        diff_obj = torch.abs(trend_g - trend_g_p)
        perct_obj = torch.sort(diff_obj)[0][:, :, int(self.drop * self.emb_dim)]
        perct_obj = perct_obj.unsqueeze(-1).repeat(1, 1, self.emb_dim)
        mask_obj = diff_obj.lt(perct_obj.to(args.device)).float()

        res_invariant, _ = self.decomp_module(x * mask_attr)
        _, trend_invariant = self.decomp_module(x * mask_obj)

        res_invariant, trend_invariant = res_invariant.permute(0,2,1), trend_invariant.permute(0,2,1)  # x: [Batch, Channel, Input length]
        res_invariant_pred = self.model_res(res_invariant)
        trend_invariant_pred = self.model_trend(trend_invariant)
        outputs = res_invariant_pred + trend_invariant_pred
        outputs = outputs.permute(0,2,1)    # x: [Batch, Input length, Channel]
        
        # 使用傅里叶变换分解季节性和趋势
        seasonal, trend = self.decompose_fourier(outputs, seasonal_ratio=0.2)

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        y = y[:, -args.pred_len:, f_dim:].to(args.device)
        pred = outputs 
        true = y  
        loss = self.criterion(pred, true) * 1/2 

        x_pred = res_p + trend_p
        x_pred = x_pred[:, -args.pred_len:, f_dim:]
        loss += self.criterion(x_pred, y)
        
        loss_res = self.criterion(res_p, seasonal)
        loss_pos_res = self.criterion(res_pp, seasonal)
        loss_trend = self.criterion(trend_p, trend)
        loss_pos_trend = self.criterion(trend_pp, trend)

        loss += (loss_res + loss_trend + loss_pos_res + loss_pos_trend) * 1/4

        res_invariant_pred, trend_invariant_pred = res_invariant_pred.permute(0,2,1), trend_invariant_pred.permute(0,2,1)
        loss_rep_res = self.criterion(res_invariant_pred, seasonal)
        loss_rep_trend = self.criterion(trend_invariant_pred, trend)

        loss += self.lambda_rep * (loss_rep_res + loss_rep_trend)

        res_grads = []
        res_env_loss = [loss_res, loss_pos_res] 
        res_network = nn.Sequential(self.decomp_module, self.model_res)
        trainable_params = [p for p in res_network.parameters() if p.requires_grad]
        for i in range(2):
            res_env_grad = autograd.grad(res_env_loss[i], trainable_params, create_graph=True)
            res_grads.append(res_env_grad)
        res_penalty_value = 0
        for i in range(len(res_grads[0])):
            res_penalty_value += (res_grads[0][i] - res_grads[1][i]).pow(2).sum()
        loss_grad_res = res_penalty_value
        

        trend_grads = []
        trend_env_loss = [loss_trend, loss_pos_trend]
        trend_network = nn.Sequential(self.decomp_module, self.model_trend)
        trainable_params = [p for p in trend_network.parameters() if p.requires_grad]
        for i in range(2):
            trend_env_grad = autograd.grad(trend_env_loss[i], trainable_params, create_graph=True) 
            trend_grads.append(trend_env_grad)
        trend_penalty_value = 0
        for i in range(len(trend_grads[0])):
            trend_penalty_value += (trend_grads[0][i] - trend_grads[1][i]).pow(2).sum()
        loss_grad_trend = trend_penalty_value

        loss += self.lambda_grad * (loss_grad_res + loss_grad_trend)

        '''
        res_init_p, trend_init_p = res_init_p.permute(0,2,1), trend_init_p.permute(0,2,1)  # x: [Batch, Input length, Channel]
        res_init_pp, trend_init_pp = res_init_pp.permute(0,2,1), trend_init_pp.permute(0,2,1)
        recon_1 = res_init_p + trend_init_p
        recon_2 = res_init_pp + trend_init_pp

        loss_rec_1 = self.criterion(recon_1, x.detach())
        loss_rec_2 = self.criterion(recon_2, x.detach())
        loss_rec = self.lambda_rec * (loss_rec_1 + loss_rec_2)

        loss += loss_rec  

        if epoch >= self.res_epoch:
            res_feat = [res_init_p,res_init_pp]
            trend_feat = [trend_init_p,trend_init_pp]
            a_y = [seasonal,seasonal]
            o_y = [trend,trend]

            a = torch.randperm(2)
            o = torch.randperm(2)

            new_res_feat = [0,0]
            new_trend_feat = [0,0]
            new_comp = [0,0]

            loss_swap_res = 0.0
            loss_swap_trend = 0.0

            for i in range(2):
                new_res_feat[i]=res_feat[a[i]]
                new_trend_feat[i]=trend_feat[o[i]]

                new_comp[i] = new_res_feat[i] + new_trend_feat[i]

                res_new, trend_new = self.decomp_module(new_comp[i])
                res_new, trend_new = res_new.permute(0,2,1), trend_new.permute(0,2,1)  # x: [Batch, Channel, Input length]
                res_new_p = self.model_res(res_new)
                trend_new_p = self.model_trend(trend_new)
                res_new_p, trend_new_p = res_new_p.permute(0,2,1), trend_new_p.permute(0,2,1)

                loss_swap_res += self.criterion(res_new_p, a_y[a[i]])
                loss_swap_trend += self.criterion(trend_new_p, o_y[o[i]])

            loss_swap = self.lambda_res*(loss_swap_res+loss_swap_trend)
            loss += loss_swap
            '''

        return loss, None

   
    def val_forward(self, x):
        res_init, trend_init = self.decomp_module(x)
        res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
        res = self.model_res(res_init)
        trend = self.model_trend(trend_init)
        x = res + trend
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return None, x

    def forward(self, args, x, y, epoch):
        if self.training:
            loss, pred = self.train_forward(args, x, y, epoch)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred


def sample(alpha,x1,x2,x3):
    if alpha<=1/3:
        return x1,0
    elif alpha<=2/3 and alpha>1/3:
        return x2,0
    else:
        return x3,1

