# Voxer: Text-to-3D Voxel Generation

基于 MAE + VQ-VAE + GPT 的文本到三维体素生成管线。

## Pipeline

```
Stage 1                Stage 2              Stage 3                Stage 4
┌─────────┐          ┌─────────┐          ┌─────────┐            ┌──────────┐
│ Objaverse│   glb    │ 3D MAE  │  768d    │ VQ-VAE  │   token    │ VoxelGPT │
│ 3D models├─────────►│ Encoder ├─────────►│ (8192)  ├───────────►│(causal)  │
│   +text  │          │  + VQ   │          │ Enc→Dec │   seq      │ text→tok │
└─────────┘          └─────────┘          └────┬─────┘            └────┬─────┘
                                               │                       │
                                         64^3 voxels              token seq
                                          (重建)                  (生成)
```

- **Stage 1** — 从 Objaverse 下载 glb 模型，转为 64³ RGBA 体素，提取文字描述
- **Stage 2** — 3D MAE 预训练：随机 mask 75% 体素块，训练 encoder/decoder 重建
- **Stage 3** — VQ-VAE：在编码器瓶颈插入 EMA codebook (8192×256)，实现体素↔离散token
- **Stage 4** — VoxelGPT：自回归 transformer + cross-attention，从文本生成 token 序列

## 项目结构

```
vox/
├── voxer_main.ipynb          # Colab 主笔记本（一键运行全流程）
├── README.md
└── voxer/                    # Python 包
    ├── __init__.py
    ├── mae.py                # 3D Masked Autoencoder (ViT encoder/decoder)
    ├── vqvae.py              # Vector Quantizer (EMA codebook) + VQ-VAE
    ├── transformer.py        # VoxelGPT: text-conditioned autoregressive model
    ├── data.py               # 数据集类、DataLoader、数据增强
    ├── train.py              # train_mae / train_vqvae / train_generator
    ├── eval.py               # 评估指标、可视化、文生体素推理
    └── utils.py              # 位置编码、随机种子、工具函数
```

## 快速开始

### Colab 运行

1. 将 `voxer/` 目录和 `voxer_main.ipynb` 上传到 Colab 的 `/content/` 目录
2. 按顺序执行 notebook 中的 5 个 Stage 代码块
3. Stage 1 中可调整 `NUM_MODELS` 控制数据量（建议 A100: 5000+，T4: 500~1000）

### 计算资源估算

| 阶段 | GPU 时间 (A100) | 说明 |
|------|----------------|------|
| Stage 1 | 2-4 h | 下载 + 体素化 |
| Stage 2 | 50-100 h | MAE 预训练 (200 epochs) |
| Stage 3 | 30-60 h | VQ-VAE 微调 (100 epochs) |
| Stage 4 | 100-200 h | GPT 训练 (300 epochs) |
| **合计** | **~200-400 h** | |

### 关键超参

| 参数 | 值 | 说明 |
|------|-----|------|
| 体素分辨率 | 64³ × 4ch (RGBA) | |
| Patch 大小 | 8³ | 共 512 个 patch |
| MAE mask ratio | 75% | Encoder 只处理可见 patch |
| MAE encoder | 12层/12头/768d | ViT-Base |
| MAE decoder | 4层/12头/384d | 轻量 decoder |
| Codebook 大小 | 8192 × 256d | EMA 更新，decay=0.99 |
| VoxelGPT | 12层/12头/768d | Causal + cross-attention |
| Text encoder | all-MiniLM-L6-v2 | 384d 输出 |

## 依赖

```
torch
objaverse
trimesh
sentence-transformers
numpy
matplotlib
tqdm
```

## 模型架构细节

### 3D MAE
- 输入体素 [B, 4, 64, 64, 64] → patchify 为 8³ 块, 每块 8³×4=2048d → linear proj → 768d
- 随机 mask 75% (仅保留 ~128 个可见 patch)
- Encoder: 12 层 ViT, 12 heads, 仅处理可见 patch
- Decoder: 4 层 ViT, 12 heads, 处理可见 + mask tokens
- Loss: MSE 仅计算被 mask 区域

### VQ-VAE
- 预训练的 MAE encoder → Linear 768→256 → VQ (8192 codes) → Linear 256→768 → MAE decoder
- Codebook 使用 EMA 更新 (decay=0.99)
- 支持 codebook usage 监控

### VoxelGPT
- 文本 384d → MLP → 768d (作为 cross-attention context)
- Learnable start token + 512 个 token position
- 12 层 causal decoder + cross-attention
- Top-k / Top-p 采样生成

## 生成示例

```python
from sentence_transformers import SentenceTransformer
from voxer.transformer import VoxelGPT
from voxer.vqvae import VoxelVQVAE
from voxer.eval import generate_and_visualize

text_model = SentenceTransformer('all-MiniLM-L6-v2')
emb = torch.from_numpy(text_model.encode(['a wooden dining chair'])).float()

results = generate_and_visualize(
    gpt_model=gpt_model,
    vqvae=vqvae_model,
    text_emb=emb,
    temperature=0.8,
    top_k=100,
    top_p=0.95,
)
```
