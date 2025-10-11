import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力（自注意力的核心计算单元）"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: 查询向量 [batch_size, num_heads, seq_len_q, head_dim]
            k: 键向量 [batch_size, num_heads, seq_len_k, head_dim]
            v: 值向量 [batch_size, num_heads, seq_len_v, head_dim]
            mask: 掩码矩阵（可选），用于屏蔽无效位置 [batch_size, 1, seq_len_q, seq_len_k]
        Returns:
            output: 注意力加权后的输出 [batch_size, num_heads, seq_len_q, head_dim]
            attn_weights: 注意力权重 [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        d_k = q.size(-1)  # 每个头的维度
        
        # 1. 计算注意力分数：Q*K^T / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_scores = attn_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # 缩放
        
        # 2. 应用掩码（如果有）：屏蔽位置的分数设为负无穷（softmax后接近0）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 3. 计算注意力权重（softmax归一化）
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_weights = self.dropout(attn_weights)  # 防止过拟合
        
        # 4. 注意力加权求和（与V相乘）
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len_q, head_dim]
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力：将输入拆分为多个头并行计算注意力，再合并结果"""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim  # 输入特征维度（对应model.dim）
        self.num_heads = num_heads  # 头的数量
        self.head_dim = dim // num_heads  # 每个头的维度（必须整除）
        
        # 验证维度合法性
        assert self.head_dim * num_heads == dim, "dim必须是num_heads的整数倍"
        
        # 1. 定义Q、K、V的线性投影层（共享权重矩阵）
        self.q_proj = nn.Linear(dim, dim)  # 输入dim→输出dim（拆分为num_heads个head_dim）
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 2. 定义输出投影层（合并多头结果）
        self.out_proj = nn.Linear(dim, dim)
        
        # 3. 缩放点积注意力实例
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: 输入序列 [batch_size, seq_len, dim]
            mask: 掩码矩阵（可选）[batch_size, seq_len, seq_len]
        Returns:
            output: 多头注意力输出 [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. 线性投影并拆分多头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)  # [batch, seq, heads, head_dim]
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch, heads, seq, head_dim]（便于并行计算）
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 2. 调整掩码维度（适配多头）：[batch, 1, seq, seq]（在头维度广播）
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, seq, seq]
        
        # 3. 计算缩放点积注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)  # [batch, heads, seq, head_dim]
        
        # 4. 合并多头结果
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq, heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.dim)  # 拼接：[batch, seq, dim]
        
        # 5. 输出投影
        output = self.out_proj(attn_output)  # [batch, seq, dim]
        output = self.dropout(output)
        
        return output


class PositionWiseFeedForward(nn.Module):
    """位置wise前馈网络：对每个位置的特征独立做线性变换"""
    def __init__(self, dim, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim_feedforward)  # 升维：dim → dim_feedforward
        self.linear2 = nn.Linear(dim_feedforward, dim)  # 降维：dim_feedforward → dim
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 激活函数（优于ReLU，在Transformer中常用）

    def forward(self, x):
        """
        Args:
            x: 输入 [batch_size, seq_len, dim]
        Returns:
            输出 [batch_size, seq_len, dim]
        """
        x = self.linear1(x)  # [batch, seq, dim_feedforward]
        x = self.activation(x)  # 非线性变换
        x = self.dropout(x)
        x = self.linear2(x)  # [batch, seq, dim]
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层：包含多头自注意力 + 前馈网络，带残差连接和层归一化"""
    def __init__(self, dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        # 1. 多头自注意力子层
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)  # 注意力后的层归一化
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. 前馈网络子层
        self.feed_forward = PositionWiseFeedForward(dim, dim_feedforward, dropout)
        self.norm2 = nn.LayerNorm(dim)  # 前馈网络后的层归一化
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: 输入序列 [batch_size, seq_len, dim]
            mask: 注意力掩码（可选）
        Returns:
            编码后的序列 [batch_size, seq_len, dim]
        """
        # 子层1：多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, mask)  # 注意力计算
        x = x + self.dropout1(attn_output)  # 残差连接（输入+注意力输出）
        x = self.norm1(x)  # 层归一化
        
        # 子层2：前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)  # 前馈网络计算
        x = x + self.dropout2(ff_output)  # 残差连接（上一步结果+前馈输出）
        x = self.norm2(x)  # 层归一化
        
        return x


class PatchEmbedding(nn.Module):
    """图像分块嵌入：将28x28图像切分为patch，转换为序列并添加位置信息"""
    def __init__(self, input_dim, dim, num_patches, patch_size):
        super().__init__()
        self.patch_size = patch_size  # 每个patch的尺寸（如4x4）
        
        # 1. patch投影：将每个patch的像素值投影到高维空间
        self.patch_proj = nn.Linear(input_dim, dim)  # input_dim = patch_size^2（如4x4=16）
        
        # 2. 类别令牌：用于最终分类（Transformer的常见 trick）
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))  # [1,1,dim]
        
        # 3. 位置嵌入：编码patch在图像中的位置信息
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # +1是因为加了class_token
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: 输入图像 [batch_size, 1, 28, 28]（1表示灰度图）
        Returns:
            嵌入后的序列 [batch_size, num_patches+1, dim]
        """
        batch_size = x.shape[0]
        
        # 步骤1：将图像切分为patch
        # 切分：[batch, 1, 28, 28] → [batch, num_patches_h, num_patches_w, patch_size, patch_size]
        x = x.unfold(2, self.patch_size, self.patch_size)  # 高度方向切分
        x = x.unfold(3, self.patch_size, self.patch_size)  # 宽度方向切分
        
        # 展平每个patch：[batch, num_patches_h, num_patches_w, patch_size*patch_size]
        num_patches_h, num_patches_w = x.shape[2], x.shape[3]
        x = x.contiguous().view(batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size)
        
        # 步骤2：patch投影到高维空间
        x = self.patch_proj(x)  # [batch, num_patches, dim]
        
        # 步骤3：添加类别令牌
        class_tokens = self.class_token.expand(batch_size, -1, -1)  # [batch, 1, dim]
        x = torch.cat([class_tokens, x], dim=1)  # [batch, num_patches+1, dim]
        
        # 步骤4：添加位置嵌入
        x = x + self.pos_embedding  # [batch, num_patches+1, dim]
        x = self.dropout(x)
        
        return x


class TransformerMNIST(nn.Module):
    """基于手写Transformer的MNIST分类模型"""
    def __init__(self, config):
        super().__init__()
        # 计算图像分块数量：(28 / patch_size)²
        self.num_patches = (config.model.image_size // config.model.patch_size) ** 2
        
        # 验证配置参数合法性
        assert config.model.image_size % config.model.patch_size == 0, \
            f"图像尺寸 {config.model.image_size} 必须能被patch尺寸 {config.model.patch_size} 整除"
        assert config.model.input_dim == config.model.patch_size **2, \
            f"input_dim {config.model.input_dim} 必须等于patch尺寸的平方 {config.model.patch_size**2}"
        
        # 1. 图像分块嵌入层
        self.patch_embedding = PatchEmbedding(
            input_dim=config.model.input_dim,
            dim=config.model.dim,
            num_patches=self.num_patches,
            patch_size=config.model.patch_size
        )
        
        # 2. 堆叠多个Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=config.model.dim,
                num_heads=config.model.num_heads,
                dim_feedforward=config.model.dim * 4,  # 前馈网络维度通常是dim的4倍
                dropout=config.model.dropout
            ) for _ in range(config.model.num_layers)
        ])
        
        # 3. 分类头（基于类别令牌输出分类结果）
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.model.dim),  # 归一化
            nn.Linear(config.model.dim, config.model.num_classes)  # 输出10个类别（0-9）
        )

    def forward(self, x):
        """
        Args:
            x: 输入图像 [batch_size, 1, 28, 28]
        Returns:
            logits: 分类得分 [batch_size, 10]
        """
        # 步骤1：图像→序列嵌入
        x = self.patch_embedding(x)  # [batch, num_patches+1, dim]
        
        # 步骤2：通过多个Transformer编码器层
        for layer in self.encoder_layers:
            x = layer(x)  # [batch, num_patches+1, dim]（每层输入输出维度不变）
        
        # 步骤3：取类别令牌的输出做分类
        class_token_output = x[:, 0, :]  # [batch, dim]（第0个位置是class_token）
        logits = self.classifier(class_token_output)  # [batch, 10]
        
        return logits
