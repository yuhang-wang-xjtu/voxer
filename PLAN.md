# Voxer 产品化改进计划

> 面向独立游戏的风格化体素资产生成工具
>
> 将研究原型升级为可用工具

---

## 一、目标定位

### 一句话

**轻量、快速、风格化的 text-to-体素 3D 资产生成器，让独立游戏开发者用自然语言生成可部署的游戏道具。**

### 用户画像

- Unity/Godot 独立游戏开发者
- MagicaVoxel 用户
- 像素风/低多边形风格游戏
- 需要快速迭代道具设计，不想从头建模

### 核心差异化

| 维度 | DreamFusion / Shap-E | Voxer |
|------|---------------------|-------|
| 目标 | 真实感 3D | 风格化游戏资产 |
| 表示 | 连续 mesh / NeRF | 离散体素 |
| 推理速度 | 分钟级 | **秒级** |
| 最低 GPU | RTX 3090+ | **T4 (免费 Colab)** |
| 模型大小 | GB 级 | **< 500 MB (量化后 < 150 MB)** |
| 输出格式 | .obj / .ply | .vox / .glb |
| 游戏引擎就绪 | 否 | **是** |

---

## 二、现状诊断

### 已完成

- [x] 数据管线：Objaverse 下载 → 64³ RGBA 体素化
- [x] 3D MAE 预训练模块
- [x] EMA-based VQ-VAE (8192 codebook)
- [x] Text-conditioned 自回归 VoxelGPT
- [x] 基础评估（IoU, MSE）
- [x] 模块化代码结构

### 待完成

- [ ] 端到端训练并产出可展示的生成结果
- [ ] 风格化生成能力
- [ ] 推理性能优化
- [ ] 游戏引擎格式导出
- [ ] 交互式使用界面
- [ ] 面向游戏的评估指标

---

## 三、数据策略（新增）

### 当前问题

| 问题 | 详情 |
|------|------|
| **存储瓶颈** | 原始 glb 文件 50-200MB/个，5000 个 = 250GB，Colab/Drive 无法存储 |
| **背景污染** | 部分 Sketchfab 模型的 glb 包含场景地面、墙壁等背景几何体，体素化后混入目标物体 |
| **2D 分析缺失** | 删掉原模型后无法用 CLIP 等 2D 模型做伪标签或质量评估 |

### 解决方案

#### 3.1 流式体素化（解决存储瓶颈）

```
流程：下载 glb → 渲染 6 视角参考图 → 体素化 → 保存体素+元数据 → 删除 glb
```

- 一次只下载一个模型，处理完立即删除原文件
- 体素文件 ~1MB/个，6 张参考图 ~240KB
- 5000 个模型从 250GB → ~6GB（40× 压缩）
- 支持断点续传（`--resume`）

**工具**：`voxer/stream_voxelize.py`

```python
from voxer.stream_voxelize import stream_voxelize_batch, load_voxels_from_stream

# 一键处理
stats = stream_voxelize_batch(
    uids=selected_uids,
    output_dir="/content/voxel_output",
    render_views=True,        # 保留 2D 参考图
    filter_background=True,   # 过滤背景几何体
    resume=True,              # 支持中断继续
)

# 加载预处理好的数据
voxels = load_voxels_from_stream("/content/voxel_output", max_models=5000)
```

#### 3.2 质量检查工作流（解决背景污染）

使用 `voxer/inspect.py` 两步检查：

**Step A: 批量自动检查**

```python
from voxer.inspect import batch_inspect

# 扫描下载目录，生成 HTML 报告
stats = batch_inspect(
    model_dir="/content/hf_models",
    output_dir="/content/inspection",
    max_models=500,  # 先查 500 个
)
# 打开 inspection/report.html 查看
```

**Step B: 单模型细查**

```bash
python -m voxer.inspect /path/to/model.glb --single -o ./check
# 输出：各子 mesh 的 bbox/体积/平整度 + 背景标记 + 3 视图对比图
```

#### 3.3 背景检测启发式规则

| 规则 | 阈值 | 说明 |
|------|------|------|
| 平整度极高 | flatness > 50 | 地面/墙面（一格远大于其他两格） |
| 体积占比过大 | > 70% 总体积 | 场景包围盒 |
| 质心偏移 | 距集群中心 > 2.5σ | 与其他物体不在同一位置 |

**在 Colab 上的使用流程**：

```
1. stream_voxelize_batch(5000 models)
   ↓ 自动过滤背景 + 删除原文件
2. 加载 voxels + manifest
   ↓ 
3. 抽取 200 个模型用 inspect 生成 HTML 报告
   ↓ 肉眼确认质量
4. 没有问题 → 全量用于训练
   有问题   → 统计问题比例，调整 filter_background 阈值
```

#### 3.4 2D 参考图用途

虽然原模型被删除，但保存了 6 视角投影图（front/back/left/right/top/bottom），可以：
- 用 CLIP 计算 image-text 相似度做伪标签精炼
- 用 CLIP image encoder 提取图像特征做风格分类
- 训练时可视化检查数据质量
- 生成 report 时作为对比

**存储占用**：6 张图 × 40KB × 5000 个 = 1.2GB，可接受。

---

## 四、改进清单

### 第一梯队：核心能力（必须有）

#### 3.1 KV-Cache 推理加速

**目标**：将自回归生成 512 token 的时间从 10-20s 降到 < 3s

**方案**：
```python
# 伪代码
class VoxelGPTWithCache(VoxelGPT):
    def generate_fast(self, text_emb):
        kv_cache = None
        x = self.start_token  # [B, 1, D]
        
        for step in range(512):
            if kv_cache is None:
                # First step: compute full attention, store K,V
                h, kv_cache = self._cached_forward(x, text_feat, step)
            else:
                # Subsequent steps: only compute new token, reuse cache
                x_new = x[:, -1:, :]  # Only last token
                h, kv_cache = self._cached_forward(x_new, text_feat, step, kv_cache)
            
            next_token = self._sample_token(h[:, -1:])
            next_emb = self.token_embed(next_token) + self.pos_embed[step + 1]
            x = torch.cat([x, next_emb], dim=1)
```

**文件**：`voxer/transformer_cache.py`

**预期效果**：

| | 无 KV-cache | 有 KV-cache |
|---|------------|------------|
| T4 推理时间 | ~18s | ~3s |
| A100 推理时间 | ~8s | ~1s |

#### 3.2 .vox 导出器

**目标**：生成的体素可直接导入 MagicaVoxel / Unity / Godot

**方案**：实现 MagicaVoxel .vox 格式写入（规范公开，约 300 行代码）

```
.vox 文件结构:
┌── RIFF Header
├── MAIN chunk
├── SIZE chunk ( x, y, z 分辨率 )
└── XYZI chunk ( 非空体素坐标 + 调色板索引 )
    └── RGBA chunk ( 调色板颜色表 )
```

**文件**：`voxer/export.py`

**验收**：生成椅子 → 双击 .vox 在 MagicaVoxel 中打开 → 可编辑

#### 3.3 颜色调色板量化

**目标**：将连续 RGB 体素映射到有限调色板，获得像素风/卡通风格

**方案**：
- 在 VQ-VAE decoder 输出后加一个可微的调色板约束损失
- 推理时后处理：对 RGB 值做 K-means 聚类到 16/32 色调色板
- 可选：提供 3 套预设调色板（像素风暖色 / 像素风冷色 / 卡通明亮）

**文件**：`voxer/stylize.py`

**预期效果**：同一个体素模型，原始 vs 调色板量化后，明显具备风格化质感。

#### 3.4 风格标签注入

**目标**：用户可以通过自然语言指定风格（"像素风"、"低多边形"、"卡通"）

**方案**：
- 训练数据提取阶段：从 Objaverse 的 name/tags 中解析风格关键词，拼接到 text 中
- Text embedding 编码整个描述（包括风格词）
- GPT 训练时学习风格-形状的联合分布

**数据准备改动**：
```python
# 原始：  "a wooden chair with armrests"
# 改为：  "voxel style: a wooden chair with armrests"
# 或：    "style: low poly, object: a wooden chair with armrests"
```

**预期效果**：输入 `"voxel style: a dragon"` vs `"smooth: a dragon"` 生成不同风格的龙。

#### 3.5 跑出可展示的生成结果

**目标**：20+ 组「文字 → 生成体素」对比图，至少 5 组包含「真实体素 vs 检索 baseline vs Voxer 生成」

**内容**：
- 10 组不同类别（椅子、桌子、灯具、沙发、箱子、树、剑、盾、头盔、药瓶）
- 5 组失败案例分析
- 3 组同一 prompt × 3 种风格对比

**产出**：`results/gallery.png` 一张汇总图 + 单张高清图

---

### 第二梯队：体验升级（大幅加分）

#### 3.6 Gradio 交互 Demo

**目标**：一个可直接访问的网页，输入文字 → 显示旋转 3D 体素 → 下载 .vox

**方案**：
- 用 Gradio `Model3D` 组件渲染体素
- 左侧：输入框 + 风格下拉框 + 随机种子滑块
- 右侧：3D 旋转区 + "Downloa  .vox" 按钮
- 部署到 HuggingFace Spaces（免费）

**文件**：`app.py`

**预期效果**：面试时打开链接直接演示，比 PPT 截图强 10 倍。

#### 3.7 条件补全（Inpainting）

**目标**："椅子的上半部分换成皇冠" — mask 部分 token，重新生成

**方案**：
- 用户输入原模型 + mask 区域（鼠标涂抹 / 坐标指定）
- 将 mask 区域对应的 token 序列段设为 `[MASK]`
- GPT 只更新被 mask 的 token，其他位置固定

**文件**：`voxer/inpaint.py`

**预期效果**：原椅子 → mask 上半部分 → "a crown" → 靠背变成皇冠的椅子。

#### 3.8 模型压缩

**目标**：模型文件 < 150MB，推理 VRAM < 3GB

| 技术 | 效果 | 实现 |
|------|------|------|
| FP16 存储 | 减半 | `model.half()` |
| INT8 量化 | 再减半 | `torch.quantization.quantize_dynamic()` |
| 蒸馏（可选） | GPT 12 层 → 6 层 | 用 student-teacher 训练小模型 |

**预期效果**：

| 指标 | 未优化 | 优化后 |
|------|--------|--------|
| 模型文件 | 880 MB | 110 MB |
| 推理 VRAM | 8 GB | 2.1 GB |
| 推理时间 (T4) | 18s | 3s (含 KV-cache) |

#### 3.9 游戏向评估指标

**目标**：量化"这个生成的体素能不能放进游戏"

| 指标 | 定义 | 阈值 |
|------|------|------|
| Sparsity | 非空体素占比 | < 20%（游戏资产不应该是实心方块） |
| Watertightness | 最大连通分量占比 | > 95%（一个整体，不是散落碎片） |
| Color count | 实际使用的颜色数量 | < 32（游戏调色板限制） |
| Symmetry | 左右镜像的 IoU | > 0.7（家具通常对称） |
| Export rate | 成功导出 .vox 的比例 | > 98% |

**文件**：`voxer/metrics.py`

---

### 第三梯队：高级功能（锦上添花）

#### 3.10 多分辨率 LOD 输出

**目标**：同一模型生成 16³ / 32³ / 64³ 三个版本，游戏内按距离切换

**方案**：VQ-VAE decoder 输出的中间层特征可解码为不同分辨率体素

#### 3.11 批量变体 + 用户选择

**目标**：同一 prompt 生成 16 个变体 → 网格展示 → 用户点选 → 下载

**方案**：调整 GPT 采样的随机种子，生成 16 个 token 序列

#### 3.12 分层 VQ（粗→细）

**目标**：先生成 4³ 粗 token（64 个），再并行生成 8³ 细 token，实现 3-5× 加速

**方案**：类似 VQ-VAE-2 的 hierarchical structure

#### 3.13 .glb Mesh 导出

**目标**：对非体素用户，用 Marching Cubes 从体素提取 mesh → 导出 .glb

#### 3.14 风格参考图

**目标**：给一张参考图，CLIP image encoder 提取 style embedding，引导生成

---

## 五、时间线

```
第 1 周 ─┬─ 3.5 跑出生成结果（现有代码，需要 GPU 时间）
        ├─ 3.4 风格标签注入（数据层改动）
        └─ 3.3 调色板量化
            ↓
第 2 周 ─┬─ 3.2 .vox 导出器
        ├─ 3.1 KV-cache
        └─ 3.9 游戏向评估指标
            ↓
第 3 周 ─┬─ 3.6 Gradio Demo
        ├─ 3.8 模型量化
        └─ 整理生成图集
            ↓
第 4 周 ─┬─ 3.7 条件补全
        ├─ 撰写项目报告
        └─ 打磨展示材料
```

---

## 六、交付物清单

```
voxer/
├── 核心（已完成）
│   ├── mae.py
│   ├── vqvae.py
│   ├── transformer.py
│   ├── data.py
│   ├── train.py
│   └── eval.py
│
├── 数据工具（已新增）
│   ├── inspect.py               ← 模型检查 + 背景检测 + HTML 报告
│   └── stream_voxelize.py       ← 流式下载→体素化→删除原文件
│
├── 新增（第一梯队）
│   ├── transformer_cache.py    ← KV-cache 加速推理
│   ├── export.py               ← .vox / .glb 导出
│   ├── stylize.py              ← 调色板量化 + 风格控制
│   └── metrics.py              ← 游戏向评估指标
│
├── 新增（第二梯队）
│   ├── app.py                  ← Gradio 交互 Demo
│   ├── inpaint.py              ← 条件补全
│   └── compress.py             ← 模型量化/蒸馏
│
├── 产物
│   ├── checkpoints/
│   │   ├── mae_best.pt
│   │   ├── vqvae_best.pt
│   │   └── generator_best.pt
│   ├── results/
│   │   ├── gallery.png          ← 20 组生成结果汇总
│   │   ├── style_comparison.png ← 同 prompt × 3 风格
│   │   └── failure_cases.png    ← 失败案例 + 分析
│   └── benchmarks/
│       ├── eval_table.csv       ← 各模型指标对比
│       └── speed_table.csv      ← 各硬件推理速度对比
│
└── 文档
    ├── README.md                ← 项目说明
    ├── PLAN.md                  ← 本文件
    └── REPORT.md                ← 项目报告（4-6 页）
```

---

## 七、评估矩阵

简历/面试中呈现的实验表格：

### 生成质量

| 方法 | IoU↑ | Sparsity | Watertightness↑ | Color Count | Export Rate |
|------|------|----------|-----------------|-------------|-------------|
| Retrieval Baseline | 1.0 | — | — | — | — |
| Voxer (无 MAE) | ? | ? | ? | ? | ? |
| Voxer (无 VQ) | ? | ? | ? | ? | ? |
| **Voxer (完整)** | ? | ? | ? | ? | ? |

### 推理性能

| 硬件 | 方法 | 时间 (s) | VRAM (GB) | 模型大小 (MB) |
|------|------|---------|-----------|---------------|
| A100 | 原始 | ~8s | ~12 GB | ~880 MB |
| A100 | +KV-cache +int8 | ~1.2s | ~2.5 GB | ~110 MB |
| T4 | 原始 | ~18s | ~10 GB | ~880 MB |
| T4 | +KV-cache +int8 | ~3.5s | ~2.1 GB | ~110 MB |

---

*最后更新: 2026-06-09*
