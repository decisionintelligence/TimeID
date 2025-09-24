export CUDA_VISIBLE_DEVICES=5

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336

root_path_name=./dataset/
source_root_path_name=./dataset/
data_path_name=ETTm2.csv
source_data_path_name=electricity.csv
model_id_name=ETTm2
data_name=ETTm2
source_data_name=Electricity


for pred_len in 336
do
    nohup python -u main.py \
      --percent 30 \
      --emb_dim 7 \
      --lr 1e-4 \
      --max_epochs 100 \
      --save_every 10000 \
      --wd 5.0e-03 \
      --drop 0.3 \
      --dropout_rate 0.1 \
      --res_epoch 20 \
      --root_path $root_path_name \
      --source_root_path $source_root_path_name \
      --data_path $data_path_name \
      --source_data_path $source_data_path_name \
      --data $data_name \
      --source_data 'custom' \
      --data_name $model_id_name \
      --source_data_name $source_data_name \
      --features M \
      --seq_len $seq_len \
      --source_seq_len $seq_len \
      --pred_len $pred_len \
      --source_pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_model_FPT 768 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --gpt_layer 6 \
      --is_gpt 1 \
      --des 'Exp' \
      --TTA_STEPS 1 \
      --IIC_PAR 1e-3 \
      --source_batch_size 256 \
      --batch_size 256 >logs/LongForecasting/$source_data_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 2>&1 &
done