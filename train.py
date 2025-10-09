import time
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser

# 导入自定义模块
from model import TransformerMNIST
from data import get_transforms, get_datasets, get_data_loaders
from utils import (
    init_distributed, create_dirs, save_checkpoint, load_checkpoint,
    init_logging, is_main_process
)


def train_one_epoch(config: DictConfig, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, distributed: bool) -> tuple[
    float, float]:
    """训练单个epoch并返回平均损失和准确率"""
    model.train()
    # 分布式采样器：确保每个epoch数据打乱
    if distributed and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        # 数据移至GPU
        device = torch.device(f"cuda:{config.dist.local_rank}" if torch.cuda.is_available() else "cpu")
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        # 按频率打印日志（仅主进程）
        if is_main_process(config) and (batch_idx + 1) % config.train.print_freq == 0:
            avg_loss = total_loss / total_samples
            acc = 100.0 * total_correct / total_samples
            elapsed_time = time.time() - start_time
            logging.info(
                f"Epoch [{epoch + 1}/{config.train.epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"| Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {elapsed_time:.2f}s"
            )
            start_time = time.time()

    # 仅在分布式模式下执行all_reduce
    if distributed:
        # 分布式聚合所有进程的统计结果
        total_loss_tensor = torch.tensor([total_loss], device=device)
        total_correct_tensor = torch.tensor([total_correct], device=device)
        total_samples_tensor = torch.tensor([total_samples], device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        # 计算全局平均指标
        avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
        avg_acc = 100.0 * total_correct_tensor.item() / total_samples_tensor.item()
    else:
        # 单卡模式直接计算
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

    return avg_loss, avg_acc


def test(config: DictConfig, model: nn.Module, test_loader: torch.utils.data.DataLoader,
         criterion: nn.Module, distributed: bool) -> tuple[float, float]:
    """测试模型并返回平均损失和准确率"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    device = torch.device(f"cuda:{config.dist.local_rank}" if torch.cuda.is_available() else "cpu")

    # 关闭梯度计算
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计指标
            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    # 仅在分布式模式下执行all_reduce
    if distributed:
        # 分布式聚合结果
        total_loss_tensor = torch.tensor([total_loss], device=device)
        total_correct_tensor = torch.tensor([total_correct], device=device)
        total_samples_tensor = torch.tensor([total_samples], device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        # 计算全局指标
        avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
        avg_acc = 100.0 * total_correct_tensor.item() / total_samples_tensor.item()
    else:
        # 单卡模式直接计算
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

    return avg_loss, avg_acc


def get_lr_scheduler(config: DictConfig, optimizer: torch.optim.Optimizer):
    """根据配置获取学习率调度器"""
    if config.lr_scheduler.type == "StepLR":
        return StepLR(
            optimizer,
            step_size=config.lr_scheduler.step_size,
            gamma=config.lr_scheduler.gamma
        )
    elif config.lr_scheduler.type == "CosineAnnealingLR":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.lr_scheduler.T_max,
            eta_min=config.train.lr * 0.01  # 最小学习率为初始值的1%
        )
    else:
        logging.warning(f"未知的学习率调度器类型: {config.lr_scheduler.type}，使用默认调度器")
        return StepLR(optimizer, step_size=10, gamma=0.1)


def main_worker(rank: int, world_size: int, config: DictConfig) -> None:
    """单个进程的训练逻辑"""
    # 1. 更新配置中的rank信息
    config.dist.rank = rank
    config.dist.world_size = world_size
    config.dist.local_rank = rank

    # 2. 初始化日志
    init_logging(config)

    # 确定是否为分布式模式
    distributed = world_size > 1
    if distributed:
        init_distributed(config)
    else:
        logging.info("单卡训练模式，不启用分布式")

    create_dirs(config)

    # 3. 加载数据
    logging.info("开始加载数据...")
    transform = get_transforms()
    train_dataset, test_dataset = get_datasets(config, transform)
    train_loader, test_loader, train_sampler = get_data_loaders(config, train_dataset, test_dataset)
    logging.info(f"数据加载完成 | 训练集大小: {len(train_dataset)} | 测试集大小: {len(test_dataset)}")

    # 4. 初始化模型、损失函数、优化器和学习率调度器
    logging.info("初始化模型和优化器...")
    device = torch.device(f"cuda:{config.dist.local_rank}" if torch.cuda.is_available() else "cpu")
    model = TransformerMNIST(config).to(device)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.dist.local_rank] if torch.cuda.is_available() else None
        )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = get_lr_scheduler(config, optimizer)  # 初始化学习率调度器

    # 5. 恢复检查点（如果需要）
    start_epoch = 0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    if config.checkpoint.resume:
        start_epoch, train_losses, train_accs, test_losses, test_accs = load_checkpoint(
            config, model, optimizer
        )
        # 恢复学习率调度器状态
        for _ in range(start_epoch):
            scheduler.step()

    # 6. 开始训练循环
    logging.info("=" * 50)
    logging.info(f"开始训练 | 总epochs: {config.train.epochs} | 起始epoch: {start_epoch + 1}")
    logging.info(f"环境: {config.environment} | 分布式模式: {'开启' if distributed else '关闭'}")
    logging.info(f"学习率调度器: {config.lr_scheduler.type}")
    logging.info("=" * 50)

    for epoch in range(start_epoch, config.train.epochs):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            config, model, train_loader, criterion, optimizer, epoch, distributed
        )

        # 测试
        test_loss, test_acc = test(config, model, test_loader, criterion, distributed)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if is_main_process(config):
            logging.info(f"当前学习率: {current_lr:.6f}")

        # 记录指标（仅主进程）
        if is_main_process(config):
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            # 打印epoch总结
            logging.info("-" * 50)
            logging.info(f"Epoch [{epoch + 1}/{config.train.epochs}] 总结:")
            logging.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logging.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            logging.info("-" * 50)

            # 保存检查点
            save_checkpoint(
                config, model, optimizer, epoch,
                train_losses, train_accs, test_losses, test_accs,
                distributed
            )

    # 7. 训练结束：保存最终模型+绘制训练曲线（仅主进程）
    if is_main_process(config):
        # 保存最终模型
        if config.train.save_model:
            final_model_path = os.path.join(config.checkpoint.checkpoint_dir, "mnist_transformer_final.pth")
            # 根据环境处理路径分隔符
            if config.environment == "local":
                final_model_path = final_model_path.replace('/', '\\')

            if distributed:
                torch.save(model.module.state_dict(), final_model_path)
            else:
                torch.save(model.state_dict(), final_model_path)

            logging.info(f"最终模型已保存至: {final_model_path}")

        # 绘制训练曲线
        if len(train_losses) > 0:
            plt.figure(figsize=(12, 4))

            # 损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
            plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", color="red")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training & Test Loss")
            plt.legend()
            plt.grid(alpha=0.3)

            # 准确率曲线
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(train_accs) + 1), train_accs, label="Train Acc", color="blue")
            plt.plot(range(1, len(test_accs) + 1), test_accs, label="Test Acc", color="red")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Training & Test Accuracy")
            plt.legend()
            plt.grid(alpha=0.3)

            # 保存曲线
            curve_path = os.path.join(config.checkpoint.checkpoint_dir, "training_curves.png")
            if config.environment == "local":
                curve_path = curve_path.replace('/', '\\')

            plt.tight_layout()
            plt.savefig(curve_path, dpi=300, bbox_inches="tight")
            logging.info(f"训练曲线已保存至: {curve_path}")

    # 8. 清理分布式环境
    if distributed:
        dist.destroy_process_group()
    logging.info(f"进程 {config.dist.rank} 训练完成")


def main():
    """主函数：解析命令行参数→加载配置→启动分布式训练"""
    # 1. 解析命令行参数
    parser = ArgumentParser(description="支持本地和K8s环境的PyTorch训练")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--num-gpus", type=int, default=None, help="指定GPU数量，默认使用所有可用GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="K8s环境下自动设置的本地rank")
    args = parser.parse_args()

    # 2. 加载配置
    config = OmegaConf.load(args.config)

    # 3. 处理K8s环境的local_rank
    if args.local_rank != -1:
        config.dist.local_rank = args.local_rank
        config.dist.rank = args.local_rank  # 在K8s中，每个进程的rank由环境变量设置

    # 4. 确定使用的GPU数量
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
    else:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # 确保不超过实际可用GPU数量
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_gpus = min(num_gpus, available_gpus)

    # 5. 根据环境自动调整配置
    if config.environment == "k8s":
        # K8s环境优化
        config.dist.backend = "nccl"  # K8s环境使用nccl后端，效率更高
        if config.data.num_workers == 0:
            config.data.num_workers = 4  # K8s环境可使用多进程数据加载
        config.data.pin_memory = True  # K8s环境开启pin_memory加速
        # 在K8s中，master_addr和master_port由PyTorchJob自动设置
        logging.info(f"K8s环境配置已应用 | 使用GPU数量: {num_gpus}")
    else:
        # 本地Windows环境
        logging.info(f"本地环境配置已应用 | 检测到 {available_gpus} 个可用GPU，将使用 {num_gpus} 个进行训练")

    # 6. 启动训练
    if num_gpus <= 1 or (config.environment == "k8s" and args.local_rank != -1):
        # 单GPU训练或K8s环境（由PyTorchJob管理进程）
        if config.environment == "k8s" and args.local_rank != -1:
            main_worker(args.local_rank, num_gpus, config)
        else:
            main_worker(0, 1, config)
    else:
        # 多GPU分布式训练（本地环境）
        # 设置启动方法为spawn（Windows唯一支持的多进程方式）
        mp.set_start_method('spawn', force=True)

        # 启动多个进程
        mp.spawn(
            main_worker,
            args=(num_gpus, config),
            nprocs=num_gpus,
            join=True
        )


# 多进程安全防护
if __name__ == "__main__":
    # 防止Windows命令行输出乱码
    import sys
    import io

    # 仅在本地环境需要设置编码
    config = OmegaConf.load("config.yaml") if os.path.exists("config.yaml") else None
    if config and config.environment == "local":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    main()
