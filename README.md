# MNIST-Transformer

基于 PyTorch 实现的 Transformer 模型用于 MNIST 手写数字分类，核心特点是无缝适配 Windows 本地开发与 Kubernetes 集群部署，支持单卡 / 多卡分布式训练，包含完整的训练、 checkpoint 管理、推理可视化流程。

## 项目亮点
多环境兼容：通过配置文件一键切换local（Windows 本地）和k8s（K8s 集群）环境，自动适配参数（如数据加载、GPU 通信后端）；

分布式训练：支持单卡 / 多卡训练，使用DistributedSampler和dist.all_reduce确保数据不重复、指标准确聚合；

工程化设计：模块化拆分（模型 / 数据 / 训练 / 工具），包含日志记录、断点续训、训练曲线可视化等功能；


## 目录结构
```
mnist-transformer/
├── config.yaml          # 全局配置文件（环境/训练/模型/数据参数）
├── model.py             # Transformer模型定义（含PatchEmbedding层）
├── data.py              # 数据加载与预处理（MNIST数据集+DistributedDataLoader）
├── train.py             # 训练主逻辑（单/多卡训练、评估、 checkpoint 保存）
├── infer.py             # 推理与可视化（模型加载、图片预测、结果展示）
├── utils.py             # 工具函数（分布式初始化、日志、 checkpoint 管理）
├── checkpoints/         # 模型 checkpoint 保存目录（自动创建）
├── logs/                # 训练日志目录（自动创建）
├── data/                # MNIST数据集目录（自动下载）
└── test_digit/          # 推理测试图片目录（需手动放入测试图）
```

## 快速开始
### 1. 环境准备
```bash
# 安装PyTorch（示例为CUDA 11.8）
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# 安装其他依赖
pip install omegaconf matplotlib pillow
```

本地环境：Windows 系统，支持单卡 / 多卡（需 NVIDIA GPU+CUDA）；

K8s 环境：需配置 PyTorchJob（如 Volcano 调度），自动获取MASTER_ADDR/RANK等环境变量。

### 2. 配置修改

修改config.yaml关键参数（根据环境调整）：

```yaml
# 1. 切换环境（local/k8s）
environment: "local"

# 2. 训练参数（本地RTX 4060 8GB建议batch_size=64，K8s可设256/512）
train:
  epochs: 10
  batch_size: 64
  lr: 0.001

# 3. 数据路径（本地默认./data，K8s设为挂载路径如/data）
data:
  data_root: "./data"
  num_workers: 0  # Windows设0，K8s可设4

# 4. 模型参数（可调整patch_size/dim等）
model:
  image_size: 28
  patch_size: 4
  dim: 32  # 显存不足可减小，K8s可增大至64
```

### 3. 启动训练
本地单卡训练

```bash
python train.py --config config.yaml
```
本地多卡训练（如 2 张 GPU）

```bash
python train.py --config config.yaml --num-gpus 2
```
K8s 集群训练（需提前配置 PyTorchJob）

```bash
# K8s会自动设置local_rank，无需指定num-gpus
python train.py --config config.yaml --local_rank $LOCAL_RANK
```

训练输出：

日志：logs/train.log（主进程写入，含训练指标）；

模型：checkpoints/下保存 epoch checkpoint 和最终模型（mnist_transformer_final.pth）；

曲线：训练结束后生成checkpoints/training_curves.png（损失 / 准确率曲线）。

### 4. 模型推理

准备测试图片：在test_digit/目录放入手写数字图片（如test_digit0.png，建议 28x28 灰度图）；

启动推理：

```
python infer.py
```

推理输出：

控制台打印预测类别和置信度；

弹出窗口显示 “原始图片” 和 “10 个类别的概率分布”（红色柱为预测类别）。

## 核心功能说明

### 1. 断点续训

修改config.yaml开启续训：

```yaml
checkpoint:
  resume: true
  resume_epoch: -1  # -1表示加载最新checkpoint，也可指定具体epoch（如5）
```

重启训练命令即可自动恢复模型权重、优化器状态和训练记录。

### 2. 学习率调度

支持两种调度器（在config.yaml切换）：

StepLR：每隔step_size个 epoch 将学习率乘以gamma；

CosineAnnealingLR：学习率按余弦曲线周期性下降。

### 3. 日志查看

控制台实时输出训练进度（仅主进程）；

日志文件logs/train.log记录完整训练过程（含时间、Rank、指标），便于问题定位。

## 许可证

本项目使用 MIT License 开源，允许个人 / 商业使用、修改、分发，需保留原作者版权声明。

## 注意事项
Windows 环境：多卡训练需确保 CUDA 版本≥11.0，且 PyTorch 支持spawn多进程；

K8s 环境：需挂载data/checkpoints/logs目录（避免数据丢失），并配置nccl通信后端；

推理图片：建议使用 28x28 灰度图，若图片尺寸不符，可在preprocess_image函数中添加 Resize 逻辑。
