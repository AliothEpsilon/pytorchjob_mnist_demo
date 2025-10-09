import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """图像分块嵌入层：将图像切分为patch并添加位置嵌入和类别令牌"""

    def __init__(self, input_dim: int, dim: int, num_patches: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size  # 保存patch大小用于计算
        self.patch_embedding = nn.Linear(input_dim, dim)  # patch投影
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))  # 类别令牌（用于分类）
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置嵌入

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1, 28, 28] → 切分为patch并展平
        batch_size = x.shape[0]

        # 使用保存的patch_size进行分块
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # 计算展平后的维度：patch_size × patch_size × 通道数(1)
        x = x.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)

        # 投影+添加类别令牌+添加位置嵌入
        x = self.patch_embedding(x)  # [batch_size, num_patches, dim]
        class_tokens = self.class_token.expand(batch_size, -1, -1)  # [batch_size, 1, dim]
        x = torch.cat([class_tokens, x], dim=1)  # [batch_size, num_patches+1, dim]
        x = x + self.position_embedding  # 位置嵌入叠加

        return x


class TransformerMNIST(nn.Module):
    """基于Transformer的MNIST分类模型"""

    def __init__(self, config):
        super().__init__()
        # 计算总patch数：(图像尺寸 / patch尺寸)²
        self.num_patches = (config.model.image_size // config.model.patch_size) ** 2

        # 验证配置参数是否匹配
        assert config.model.image_size % config.model.patch_size == 0, \
            f"image_size ({config.model.image_size}) 必须能被 patch_size ({config.model.patch_size}) 整除"
        assert config.model.input_dim == config.model.patch_size ** 2, \
            f"input_dim ({config.model.input_dim}) 必须等于 patch_size 的平方 ({config.model.patch_size ** 2})"

        # 组件初始化
        self.patch_embedding = PatchEmbedding(
            input_dim=config.model.input_dim,
            dim=config.model.dim,
            num_patches=self.num_patches,
            patch_size=config.model.patch_size  # 传递patch_size参数
        )

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.dim,
            nhead=config.model.num_heads,
            dim_feedforward=config.model.dim * 4,  # 前馈网络隐藏层维度
            dropout=config.model.dropout,
            batch_first=True  # 输入格式：[batch_size, seq_len, dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.num_layers
        )

        # 分类头（基于类别令牌输出）
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.model.dim),  # 层归一化
            nn.Linear(config.model.dim, config.model.num_classes)  # 分类输出
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1, 28, 28] → 输出: [batch_size, num_classes]
        x = self.patch_embedding(x)  # [batch_size, num_patches+1, dim]
        x = self.transformer_encoder(x)  # [batch_size, num_patches+1, dim]
        class_token = x[:, 0, :]  # 取类别令牌（第0个位置）：[batch_size, dim]
        output = self.classifier(class_token)  # [batch_size, num_classes]

        return output
