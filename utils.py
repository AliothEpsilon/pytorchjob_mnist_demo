import os
import json
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omegaconf import DictConfig


def init_distributed(config: DictConfig) -> None:
    """初始化分布式训练环境（兼容不同环境）"""
    # 对于K8s环境，master_addr和master_port通常由环境变量设置
    if config.environment == "k8s":
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', config.dist.master_addr)
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', config.dist.master_port)
        os.environ['RANK'] = os.environ.get('RANK', str(config.dist.rank))
        os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', str(config.dist.world_size))
    else:
        # 本地环境
        os.environ['MASTER_ADDR'] = config.dist.master_addr
        os.environ['MASTER_PORT'] = config.dist.master_port

    # 构建初始化参数
    init_kwargs = {
        'backend': config.dist.backend,
        'rank': int(os.environ.get('RANK', config.dist.rank)),
        'world_size': int(os.environ.get('WORLD_SIZE', config.dist.world_size)),
    }

    # 对于K8s环境，使用环境变量提供的初始化方法
    if config.environment == "k8s":
        init_kwargs['init_method'] = 'env://'
    else:
        init_kwargs['init_method'] = f"tcp://{config.dist.master_addr}:{config.dist.master_port}"

    # 初始化进程组
    dist.init_process_group(**init_kwargs)

    # 分配GPU设备
    device_id = config.dist.local_rank % torch.cuda.device_count() if torch.cuda.is_available() else -1
    torch.cuda.set_device(device_id)
    logging.info(
        f"分布式进程 {config.dist.rank} 绑定设备: {torch.cuda.get_device_name(device_id) if device_id != -1 else 'CPU'}")


def create_dirs(config: DictConfig) -> None:
    """创建日志目录、检查点目录（仅主进程执行）"""
    if is_main_process(config):
        # 根据环境处理路径
        log_dir = config.log.log_dir
        checkpoint_dir = config.checkpoint.checkpoint_dir

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"已创建目录：日志目录={log_dir}, 检查点目录={checkpoint_dir}")


def is_main_process(config: DictConfig) -> bool:
    """判断当前进程是否为主进程（rank=0）"""
    return config.dist.rank == 0 or int(os.environ.get('RANK', 0)) == 0


def save_checkpoint(config: DictConfig, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, train_losses: list, train_accs: list, test_losses: list, test_accs: list,
                    distributed: bool) -> None:
    """保存训练检查点（适配单卡/多卡模式）"""
    if not is_main_process(config):
        return

    # 检查是否达到保存频率
    if (epoch + 1) % config.train.save_freq != 0:
        return

    # 构建检查点路径
    checkpoint_path = os.path.join(config.checkpoint.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
    # 根据环境处理路径分隔符
    if config.environment == "local":
        checkpoint_path = checkpoint_path.replace('/', '\\')

    # 根据是否分布式模式获取模型状态
    if isinstance(model, DistributedDataParallel):
        # 分布式模式：模型被DDP包装，需要通过model.module获取原始模型
        model_state = model.module.state_dict()
    else:
        # 单卡模式：直接获取模型状态
        model_state = model.state_dict()

    # 保存内容
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "lr": optimizer.param_groups[0]["lr"]
    }

    torch.save(checkpoint, checkpoint_path)
    logging.info(f"检查点已保存至: {checkpoint_path}")

    # 保存最新检查点信息
    latest_info = {
        "latest_epoch": epoch + 1,
        "checkpoint_path": checkpoint_path
    }
    latest_path = os.path.join(config.checkpoint.checkpoint_dir, "latest_checkpoint.json")
    if config.environment == "local":
        latest_path = latest_path.replace('/', '\\')

    with open(latest_path, "w") as f:
        json.dump(latest_info, f, indent=2)


def load_checkpoint(config: DictConfig, model: nn.Module, optimizer: torch.optim.Optimizer):
    """加载检查点（恢复模型、优化器状态和训练记录），修复单卡兼容问题"""
    # 确定要加载的检查点路径
    if config.checkpoint.resume_epoch != -1:
        # 加载指定epoch
        checkpoint_path = os.path.join(
            config.checkpoint.checkpoint_dir,
            f"checkpoint_epoch_{config.checkpoint.resume_epoch}.pth"
        )
    else:
        # 加载最新检查点
        latest_path = os.path.join(config.checkpoint.checkpoint_dir, "latest_checkpoint.json")
        if config.environment == "local":
            latest_path = latest_path.replace('/', '\\')

        if not os.path.exists(latest_path):
            logging.warning("未找到最新检查点信息，将从头开始训练")
            return 0, [], [], [], []

        with open(latest_path, "r") as f:
            latest_info = json.load(f)
            checkpoint_path = latest_info["checkpoint_path"]

    # 检查文件是否存在
    if config.environment == "local":
        checkpoint_path = checkpoint_path.replace('/', '\\')

    if not os.path.exists(checkpoint_path):
        logging.warning(f"检查点 {checkpoint_path} 不存在，将从头开始训练")
        return 0, [], [], [], []

    # 加载检查点（映射到当前设备）
    device = torch.device(f"cuda:{config.dist.local_rank}" if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 恢复模型状态 - 修复单卡/分布式兼容问题
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 恢复学习率
    for param_group in optimizer.param_groups:
        param_group["lr"] = checkpoint["lr"]

    logging.info(f"已从检查点 {checkpoint_path} 恢复训练，将从epoch {checkpoint['epoch']} 开始")

    # 返回恢复的训练状态
    return (
        checkpoint["epoch"],
        checkpoint["train_losses"],
        checkpoint["train_accs"],
        checkpoint["test_losses"],
        checkpoint["test_accs"]
    )


def init_logging(config: DictConfig) -> None:
    """初始化日志系统（确保日志目录存在）"""
    # 确保日志目录存在，不存在则创建
    log_dir = config.log.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # 日志格式：时间-进程号-级别-信息
    log_format = "%(asctime)s - [Rank %(process)d] - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 日志级别映射
    log_level = getattr(logging, config.log.log_level.upper(), logging.INFO)

    # 初始化日志器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []  # 清空默认处理器

    # 1. 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    # 2. 文件处理器（仅主进程写入）
    if is_main_process(config):
        log_file_path = os.path.join(log_dir, config.log.log_file)
        # 确保路径使用正确的分隔符
        if config.environment == "local":
            log_file_path = log_file_path.replace('/', '\\')

        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)
        logging.info(f"日志文件已创建至: {log_file_path}")
