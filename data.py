import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from omegaconf import DictConfig


def get_transforms() -> transforms.Compose:
    """获取MNIST数据预处理流水线（归一化使用MNIST官方统计值）"""
    return transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # 标准化（均值、标准差）
    ])


def get_datasets(config: DictConfig, transform: transforms.Compose):
    """加载MNIST训练集和测试集"""
    train_dataset = datasets.MNIST(
        root=config.data.data_root,
        train=True,
        transform=transform,
        download=True  # 不存在则自动下载
    )

    test_dataset = datasets.MNIST(
        root=config.data.data_root,
        train=False,
        transform=transform,
        download=True
    )

    return train_dataset, test_dataset


def get_data_loaders(config: DictConfig, train_dataset, test_dataset):
    """创建分布式DataLoader（训练集用DistributedSampler，测试集用普通采样）"""
    # 训练集：分布式采样（确保各进程数据不重复）
    train_sampler = DistributedSampler(train_dataset) if config.dist.world_size > 1 else None

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # 分布式时关闭shuffle（由sampler控制）
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,  # 丢弃最后一个不完整的batch
        # Windows多进程数据加载特殊处理
        multiprocessing_context=None if config.data.num_workers == 0 else 'spawn'
    )

    # 测试集：无分布式采样（各进程加载完整测试集，最后聚合结果）
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
        multiprocessing_context=None if config.data.num_workers == 0 else 'spawn'
    )

    return train_loader, test_loader, train_sampler
