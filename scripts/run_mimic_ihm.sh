#!/bin/bash

model_name=MIMIC_TimeLLM
train_epochs=50
learning_rate=0.001
llama_layers=6

master_port=00097
num_process=1  # 根据可用GPU数量调整
batch_size=32
d_model=16
d_ff=32

comment='mimic_ihm_48h'

# 运行MIMIC-IHM (48小时院内死亡预测)任务
# --use_wandb --wandb_project "MIMIC-TimeLLM" --wandb_entity "hcy50662-national-university-of-singapore-students-union" \
python run_mimic.py \
  --task_name classification \
  --use_wandb --wandb_project "TimeLLM" --wandb_entity "hcy50662-national-university-of-singapore-students-union" \
  --num_workers 0 \
  --is_training 1 \
  --root_path /home/ubuntu/Virginia/output_mimic3/ihm \
  --model_id mimic \
  --model $model_name \
  --data MIMIC \
  --task ihm \
  --features M \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 17 \
  --dec_in 17 \
  --c_out 1 \
  --batch_size $batch_size \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --patience 5 \
  --llm_model BERT_TINY \
  --model_comment $comment \
  --test_mode \
  --disable_distributed

