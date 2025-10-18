#!/bin/bash
cd /home/bluestar/research/DualAnoDiff/bcm-dual-interrelated_diff
source /home/bluestar/miniconda3/etc/profile.d/conda.sh
conda activate /home/bluestar/research/DualAnoDiff/.env

CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora_single_background.py \
  --pretrained_model_name_or_path="/home/bluestar/research/DualAnoDiff/stable-diffusion-v1-5" \
  --instance_data_dir="/home/bluestar/research/DualAnoDiff/mvtec_anomaly_detection/toothbrush/test/defective" \
  --output_dir="all_generate/toothbrush/defective" \
  --instance_prompt="a srw" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --rank=32 \
  --train_text_encoder
