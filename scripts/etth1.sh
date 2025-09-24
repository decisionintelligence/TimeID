export CUDA_VISIBLE_DEVICES=3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1


for pred_len in 336
do
    nohup python -u train.py \
      --emb_dim 7 \
      --lr 1e-3 \
      --max_epochs 100 \
      --save_every 10000 \
      --wd 5.0e-05 \
      --drop 0.3 \
      --dropout_rate 0.3 \
      --res_epoch 20 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --data $data_name \
      --data_name $model_id_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --batch_size 256 >logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log 2>&1 &

done
