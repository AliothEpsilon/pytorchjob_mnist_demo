import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from model import TransformerMNIST  # 导入训练时定义的模型
from data import get_transforms  # 复用训练时的预处理逻辑


def load_trained_model(config_path: str, model_path: str) -> nn.Module:
    """
    加载训练好的模型（添加weights_only=True消除安全警告）
    """
    # 加载配置文件
    config = OmegaConf.load(config_path)

    # 初始化模型（与训练时一致）
    model = TransformerMNIST(config)

    # 加载训练好的权重：添加weights_only=True，只加载张量权重，拒绝执行代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # 关键修改

    # 切换到评估模式（关闭Dropout等训练特有的层）
    model.eval()
    model.to(device)

    print(f"模型已加载至 {device}，路径：{model_path}")
    return model, device, config


def preprocess_image(image_path: str, transform) -> torch.Tensor:
    """
    图片预处理：转为28x28灰度图→归一化→添加batch维度
    :param image_path: 输入图片路径（支持PNG/JPG等格式）
    :param transform: 训练时使用的预处理流水线
    :return: 预处理后的张量（shape: [1, 1, 28, 28]）
    """
    # 1. 读取图片（转为灰度图）
    img = Image.open(image_path).convert("L")  # "L"表示灰度模式

    # 2. 应用训练时的预处理（归一化等）
    img_tensor = transform(img)

    # 3. 添加batch维度（模型要求输入为[batch_size, 1, 28, 28]）
    img_tensor = img_tensor.unsqueeze(0)  # 从[1,28,28]→[1,1,28,28]

    return img_tensor, img  # 返回张量和原始PIL图片（用于可视化）


def predict_image(model: nn.Module, img_tensor: torch.Tensor, device: torch.device) -> tuple[int, float, np.ndarray]:
    """
    模型推理：预测图片类别
    :param model: 加载好的模型
    :param img_tensor: 预处理后的图片张量
    :param device: 运行设备（CPU/GPU）
    :return: 预测类别（0-9）、最大概率、所有类别的概率数组
    """
    # 关闭梯度计算（推理时不需要）
    with torch.no_grad():
        # 图片张量移至设备
        img_tensor = img_tensor.to(device)

        # 模型输出（shape: [1, 10]，10个类别的logits）
        outputs = model(img_tensor)

        # 转为概率（softmax）
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # [10]

        # 取概率最大的类别和概率值
        pred_class = np.argmax(probabilities)
        pred_prob = probabilities[pred_class]

    return pred_class, pred_prob, probabilities


def visualize_result(img: Image.Image, pred_class: int, pred_prob: float, probabilities: np.ndarray):
    """可视化：显示原始图片和预测结果（仅用Windows自带字体，消除警告）"""
    # --------------------------
    # 关键修改：只保留Windows系统自带的中文字体
    # --------------------------
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]  # 这两个字体Windows必装
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

    # 创建2个子图（图片+概率分布）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1：显示原始图片
    ax1.imshow(img, cmap="gray")
    ax1.set_title(f"预测结果：手写数字 {pred_class}\n置信度：{pred_prob:.4f}", fontsize=14)
    ax1.axis("off")

    # 子图2：显示所有类别的概率分布
    classes = list(range(10))
    bars = ax2.bar(classes, probabilities, color="skyblue")
    bars[pred_class].set_color("red")
    ax2.set_xlabel("数字类别（0-9）", fontsize=12)
    ax2.set_ylabel("预测概率", fontsize=12)
    ax2.set_title("模型对所有类别的置信度分布", fontsize=14)
    ax2.set_xticks(classes)
    ax2.set_ylim(0, 1.05)

    # 在每个柱子上标注概率值
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f"{prob:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()


def main():
    # --------------------------
    # 1. 配置参数（根据实际路径修改）
    # --------------------------
    config_path = "config.yaml"  # 训练时的配置文件
    model_path = "checkpoints/mnist_transformer_final.pth"  # 训练好的模型
    image_path = "test_digit/test_digit0.png"  # 待预测的图片（自己准备）

    # --------------------------
    # 2. 加载模型和预处理流水线
    # --------------------------
    model, device, config = load_trained_model(config_path, model_path)
    transform = get_transforms()  # 复用训练时的预处理（保证一致性）

    # --------------------------
    # 3. 图片预处理
    # --------------------------
    img_tensor, raw_img = preprocess_image(image_path, transform)
    print(f"图片预处理完成，张量形状：{img_tensor.shape}")

    # --------------------------
    # 4. 模型预测
    # --------------------------
    pred_class, pred_prob, probabilities = predict_image(model, img_tensor, device)
    print(f"预测结果：数字 {pred_class}，置信度：{pred_prob:.4f}")

    # --------------------------
    # 5. 可视化结果
    # --------------------------
    visualize_result(raw_img, pred_class, pred_prob, probabilities)


if __name__ == "__main__":
    main()