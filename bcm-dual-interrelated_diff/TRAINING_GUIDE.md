# DualAnoDiff 訓練指南

本指南記錄了在新環境中設置和運行 DualAnoDiff 訓練的完整步驟，包括所有遇到的問題和解決方案。

## 目錄

- [系統需求](#系統需求)
- [環境設置](#環境設置)
- [代碼修改](#代碼修改)
- [訓練執行](#訓練執行)
- [問題排查](#問題排查)
- [監控訓練](#監控訓練)

---

## 系統需求

### 硬件要求
- **GPU**: NVIDIA GPU with >= 20GB VRAM (測試使用 RTX A4500)
- **RAM**: >= 32GB
- **存儲**: >= 100GB 可用空間

### 軟件要求
- **OS**: Linux (測試於 Ubuntu with kernel 6.8.0-64)
- **CUDA**: 11.8
- **Python**: 3.10 (不要使用 3.13，存在兼容性問題)
- **Conda**: Miniconda 或 Anaconda

---

## 環境設置

### 1. 創建 Conda 環境

```bash
# 進入項目目錄
cd /path/to/DualAnoDiff

# 創建 Python 3.10 環境（重要：不要使用 3.13）
conda create -p ./.env python=3.10 -y
conda activate ./.env
```

### 2. 修復 requirements.txt

原始的 `requirements.txt` 存在格式錯誤，需要修復：

```bash
# 編輯 requirements.txt，修復以下兩行：
# 第 9 行：opencv-python-headless==4.7.0.72  (原為 .7.0.72)
# 第 18 行：tensorboard==2.15.0  (原為 .15.0)
```

修復後的 requirements.txt：
```
accelerate==0.24.1
clip
Cython==0.29.35
matplotlib==3.8.0
numpy==1.24.3
open-clip-torch==2.23.0
opencv-python==4.7.0.72
opencv-python-headless==4.7.0.72  # 已修復
opencv-python-headless==4.7.0.72
pandas==2.0.3
Pillow==9.4.0
pytorch-lightning==1.5.0
PyYAML==6.0
scikit-image==0.22.0
scikit-learn==1.3.2
scipy==1.10.1
setuptools==65.6.3
tensorboard==2.15.0  # 已修復
timm==0.4.12
torch==2.0.1+cu118
torchaudio==2.0.2+cu118
torchmetrics==0.6.0
torchvision==0.15.2+cu118
transformers==4.30.2
```

### 3. 安裝依賴

**重要：必須按以下順序安裝，並進行 PyTorch 升級**

```bash
# Step 1: 安裝基礎依賴
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

# Step 2: 安裝缺失的依賴
pip install einops
pip install "huggingface_hub<0.20"  # 版本 0.19.4
pip install ipdb
pip install diffusers

# Step 3: 關鍵步驟 - 升級 PyTorch 到 2.2.0
# 這是解決混合精度訓練 dtype 錯誤的關鍵
pip install --upgrade torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Step 4: 升級 xformers 以匹配 PyTorch 2.2
pip install --upgrade xformers==0.0.24
```

### 4. 配置 Accelerate

```bash
# 創建 accelerate 配置目錄
mkdir -p ~/.cache/huggingface/accelerate

# 創建配置文件
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: 'no'  # 重要：使用 fp32 訓練
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
EOF
```

### 5. 準備數據集

```bash
# 下載並解壓 MVTec AD 數據集到：
# /path/to/mvtec_anomaly_detection/
#
# 目錄結構應為：
# mvtec_anomaly_detection/
# ├── bottle/
# ├── cable/
# ├── toothbrush/
# │   ├── train/
# │   │   └── good/
# │   └── test/
# │       ├── good/
# │       └── defective/
# └── ...
```

---

## 代碼修改

### 修改 1: 更新路徑配置

**文件**: `run_mvtec_split_background_control_scale.py`

```python
# 第 9-11 行：更新為絕對路徑
export MODEL_NAME="/path/to/stable-diffusion-v1-5"
export INSTANCE_DIR="/path/to/mvtec_anomaly_detection/{name}/test/{anomaly}"
export OUTPUT_DIR="all_generate/{name}/{anomaly}"

# 第 53 行：更新 MVTec 數據集路徑
for anomaly in os.listdir(os.path.join('/path/to/mvtec_anomaly_detection',name,'test')):
```

### 修改 2: 修復混合精度問題（關鍵修改）

**文件**: `train_dreambooth_lora_single_background.py`

在第 1060 行後（`unet.set_attn_processor(unet_lora_attn_procs)` 之後）添加以下代碼：

```python
unet.set_attn_processor(unet_lora_attn_procs)
unet_lora_layers = AttnProcsLayers(unet.attn_processors)

# ========== 添加以下代碼 ==========
# Move temporal layers in attention processors to weight_dtype
for attn_processor in unet.attn_processors.values():
    if hasattr(attn_processor, 'chained_proc'):
        # For ReferenceOnlyAttnProc wrappers
        if hasattr(attn_processor.chained_proc, 'temporal_n'):
            attn_processor.chained_proc.temporal_n.to(dtype=weight_dtype)
            attn_processor.chained_proc.temporal_i.to(dtype=weight_dtype)
            attn_processor.chained_proc.temporal_q.to(dtype=weight_dtype)
            attn_processor.chained_proc.temporal_k.to(dtype=weight_dtype)
            attn_processor.chained_proc.temporal_v.to(dtype=weight_dtype)
            attn_processor.chained_proc.temporal_o.to(dtype=weight_dtype)
            attn_processor.chained_proc.temporal_condition_mlp.to(dtype=weight_dtype)
    elif hasattr(attn_processor, 'temporal_n'):
        # For direct LoRAAttnProcessor2_0 instances
        attn_processor.temporal_n.to(dtype=weight_dtype)
        attn_processor.temporal_i.to(dtype=weight_dtype)
        attn_processor.temporal_q.to(dtype=weight_dtype)
        attn_processor.temporal_k.to(dtype=weight_dtype)
        attn_processor.temporal_v.to(dtype=weight_dtype)
        attn_processor.temporal_o.to(dtype=weight_dtype)
        attn_processor.temporal_condition_mlp.to(dtype=weight_dtype)
# ========== 結束添加 ==========

# The text encoder comes from 🤗 transformers, so we cannot directly modify it.
```

### 修改 3: 簡化 LoRA 代碼

**文件**: `diffusers/models/attention_processor.py`

找到 `LoRAAttnProcessor2_0.__call__` 方法中的 LoRA 計算部分（約 1370-1413 行），替換為：

```python
# 原始代碼有複雜的 dtype 轉換和 autocast 處理
# 替換為簡化版本：

# LoRA query
query = attn.to_q(hidden_states) + scale * self.to_q_lora[i](hidden_states)

# self:
if 'condition_encoder_state' in kwargs and i == 0:
    condition_encoder_state = kwargs['condition_encoder_state']
    # Blend condition encoder state with encoder hidden states
    condition_encoder_state_processed = self.temporal_condition_mlp(condition_encoder_state)
    blended = (1 - self.temporal_a) * encoder_hidden_states + self.temporal_a * condition_encoder_state_processed
    encoder_hidden_states_for_kv = blended
else:
    encoder_hidden_states_for_kv = encoder_hidden_states

# LoRA key/value
key_input = encoder_hidden_states_for_kv if encoder_hidden_states_for_kv is not None else hidden_states
value_input = encoder_hidden_states_for_kv if encoder_hidden_states_for_kv is not None else hidden_states
key = attn.to_k(key_input) + scale * self.to_k_lora[i](key_input)
value = attn.to_v(value_input) + scale * self.to_v_lora[i](value_input)
```

同時在約 1452 行，將：
```python
x = hidden_states.half()  # Force fp16 for temporal processing in mixed precision
```
改為：
```python
x = hidden_states  # Keep original dtype (fp32 when mixed precision disabled, fp16 when enabled)
```

---

## 訓練執行

### 創建訓練腳本

```bash
cd /path/to/DualAnoDiff/bcm-dual-interrelated_diff

cat > start_training.sh << 'EOF'
#!/bin/bash
cd /path/to/DualAnoDiff/bcm-dual-interrelated_diff
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/DualAnoDiff/.env

CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_lora_single_background.py \
  --pretrained_model_name_or_path="/path/to/stable-diffusion-v1-5" \
  --instance_data_dir="/path/to/mvtec_anomaly_detection/toothbrush/test/defective" \
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
EOF

chmod +x start_training.sh
```

### 啟動訓練

```bash
# 使用 nohup 在後台運行
nohup bash start_training.sh > nohup_train_toothbrush_defective.log 2>&1 &

# 記下 PID
echo $!
```

### 訓練參數說明

- `--train_batch_size=2`: 每個 GPU 的 batch size（20GB VRAM 可用 2）
- `--max_train_steps=5000`: 總訓練步數
- `--rank=32`: LoRA 秩
- `--train_text_encoder`: 同時訓練文本編碼器
- `--resolution=512`: 圖像分辨率

**預計訓練時間**: 約 4-5 小時（RTX A4500, batch_size=2）

---

## 問題排查

### 問題 1: Python 3.13 兼容性錯誤

**錯誤**:
```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

**解決方案**: 使用 Python 3.10
```bash
conda create -p ./.env python=3.10 -y
```

### 問題 2: 混合精度 dtype 不匹配

**錯誤**:
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float
RuntimeError: expected scalar type Half but found Float
```

**原因**: PyTorch 2.0.x 在自定義 attention processors 中的混合精度訓練存在 bug

**解決方案**:
1. 升級 PyTorch 到 2.2.0
2. 使用 fp32 訓練（`mixed_precision: 'no'`）
3. 添加代碼將 temporal 層移動到正確的 dtype（見代碼修改 2）

### 問題 3: xformers 版本不匹配

**錯誤**:
```
xformers 0.0.17 requires torch==2.0.0, but you have torch 2.2.0
```

**解決方案**:
```bash
pip install --upgrade xformers==0.0.24
```

### 問題 4: huggingface_hub 兼容性

**錯誤**:
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

**解決方案**:
```bash
pip install "huggingface_hub<0.20"
```

### 問題 5: nohup 日誌為空

**原因**: 使用 `conda run` 包裹導致輸出重定向失敗

**解決方案**: 創建獨立的 bash 腳本並激活環境
```bash
# 在腳本中使用：
source /path/to/conda.sh
conda activate /path/to/.env
# 而不是：
conda run -p /path/to/.env command
```

---

## 監控訓練

### 檢查訓練進程

```bash
# 查看進程
ps aux | grep train_dreambooth_lora_single_background | grep -v grep

# 查看 GPU 使用
nvidia-smi

# 持續監控 GPU
watch -n 1 nvidia-smi
```

### 查看訓練日誌

```bash
# 實時查看日誌
tail -f nohup_train_toothbrush_defective.log

# 查看最新 30 行
tail -n 30 nohup_train_toothbrush_defective.log

# 查看最近的 loss 值
grep "loss=" nohup_train_toothbrush_defective.log | tail -20

# 查看當前訓練步數
grep "Steps:" nohup_train_toothbrush_defective.log | tail -1
```

### 訓練進度示例

```
Steps:  10%|█         | 500/5000 [10:25<1:33:45,  1.25s/it, loss=0.0234, lr=2e-5]
```

解讀：
- 完成 500/5000 步（10%）
- 已用時間：10 分 25 秒
- 預計剩餘時間：1 小時 33 分 45 秒
- 當前速度：1.25 秒/步
- 當前 loss：0.0234

### 停止訓練

```bash
# 找到進程 PID
ps aux | grep train_dreambooth_lora_single_background | grep -v grep

# 使用 PID 停止
kill <PID>

# 或強制停止
kill -9 <PID>
```

---

## 訓練輸出

### 模型權重

訓練完成後，LoRA 權重保存在：
```
all_generate/toothbrush/defective/pytorch_lora_weights.bin
```

### 推理使用

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "/path/to/stable-diffusion-v1-5",
    safety_checker=None
).to("cuda")

pipe.load_lora_weights('./all_generate/toothbrush/defective')

# 生成圖像...
```

---

## 常見配置

### 多 GPU 訓練

修改 `~/.cache/huggingface/accelerate/default_config.yaml`:

```yaml
distributed_type: 'MULTI_GPU'
num_processes: 2  # GPU 數量
```

然後使用：
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ...
```

### 調整顯存使用

如果顯存不足，調整以下參數：

```bash
--train_batch_size=1  # 減小 batch size
--gradient_accumulation_steps=2  # 增加累積步數以補償
--gradient_checkpointing  # 啟用梯度檢查點（降低顯存使用）
```

### 使用混合精度（如果 PyTorch >= 2.2）

修改 accelerate 配置：
```yaml
mixed_precision: fp16  # 或 bf16（需要 Ampere+ GPU）
```

---

## 版本信息

成功測試的環境配置：

- **OS**: Linux 6.8.0-64-generic
- **Python**: 3.10
- **PyTorch**: 2.2.0+cu118
- **torchvision**: 0.17.0+cu118
- **torchaudio**: 2.2.0+cu118
- **xformers**: 0.0.24
- **accelerate**: 0.24.1
- **transformers**: 4.30.2
- **diffusers**: latest
- **CUDA**: 11.8
- **cuDNN**: 8.7.0.84

---

## 參考資料

- [DualAnoDiff GitHub](https://github.com/yinyjin/DualAnoDiff)
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

## 故障排除檢查清單

訓練前檢查：

- [ ] Python 版本 = 3.10（不是 3.13）
- [ ] PyTorch 版本 >= 2.2.0
- [ ] xformers 版本 = 0.0.24
- [ ] accelerate 配置正確（`mixed_precision: 'no'`）
- [ ] 所有路徑更新為絕對路徑
- [ ] 代碼修改已應用（train_dreambooth_lora_single_background.py 和 attention_processor.py）
- [ ] MVTec 數據集已下載並放置在正確位置
- [ ] Stable Diffusion v1.5 模型已下載
- [ ] GPU 顯存 >= 16GB（batch_size=1）或 >= 20GB（batch_size=2）

---

**最後更新**: 2025-10-18
**維護者**: Claude Code Assistant
