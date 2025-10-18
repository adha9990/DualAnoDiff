
cd /home/bluestar/research/DualAnoDiff/bcm-dual-interrelated_diff


export MODEL_NAME="/home/bluestar/research/DualAnoDiff/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/bluestar/research/DualAnoDiff/mvtec_anomaly_detection/toothbrush/test/defective"
export OUTPUT_DIR="all_generate/toothbrush/defective"

CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora_single_background.py   --pretrained_model_name_or_path=$MODEL_NAME    --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR   --instance_prompt="a srw"   --resolution=512   --train_batch_size=2   --gradient_accumulation_steps=1   --learning_rate=2e-5   --lr_scheduler="constant"   --lr_warmup_steps=0   --max_train_steps=5000   --rank 32   --train_text_encoder
  


cd /home/bluestar/research/DualAnoDiff/bcm-dual-interrelated_diff

CUDA_VISIBLE_DEVICES=0 python inference_test_tempt.py toothbrush defective
sleep 2m


cd /home/bluestar/research/U-2-Net-master

CUDA_VISIBLE_DEVICES=0 python u2net_test.py /home/bluestar/research/DualAnoDiff/bcm-dual-interrelated_diff/generate_data/toothbrush/defective
sleep 1m

