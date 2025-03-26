import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    """
    位置编码模块：实现标准的正弦-余弦位置编码
    
    参数:
        d_model: 模型维度，即嵌入向量的维度
        max_len: 支持的最大序列长度，默认5000
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 在对数空间中一次性计算位置编码
        pe = torch.zeros(max_len, d_model).float()  # shape: [max_len, d_model]
        pe.require_grad = False  # 位置编码不需要梯度

        position = torch.arange(0, max_len).float().unsqueeze(1)  # shape: [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()  # shape: [d_model/2]

        # 使用正弦和余弦函数交替填充位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦

        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]，添加批次维度
        self.register_buffer('pe', pe)  # 将位置编码注册为缓冲区，不作为模型参数

    def forward(self, x):
        """
        输入:
            x: shape [batch_size, seq_len, feature_size]
        返回:
            位置编码: shape [1, seq_len, d_model]
        """
        return self.pe[:, :x.size(1)]  # 只返回与输入序列长度匹配的部分位置编码


class TokenEmbedding(nn.Module):
    """
    令牌嵌入模块：通过1D卷积将输入特征转换为嵌入向量
    
    参数:
        c_in: 输入通道数，即输入特征的维度
        d_model: 输出的嵌入维度
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2  # 根据PyTorch版本设置填充
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        输入:
            x: shape [batch_size, seq_len, c_in]
        返回:
            嵌入向量: shape [batch_size, seq_len, d_model]
        """
        # 将输入转置为[batch_size, c_in, seq_len]进行卷积，然后再转置回来
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # shape: [batch_size, seq_len, d_model]
        return x


class FixedEmbedding(nn.Module):
    """
    固定嵌入模块：为输入创建固定的嵌入向量，不参与梯度更新
    
    参数:
        c_in: 输入类别的数量
        d_model: 嵌入维度
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()  # shape: [c_in, d_model]
        w.require_grad = False  # 不需要梯度

        position = torch.arange(0, c_in).float().unsqueeze(1)  # shape: [c_in, 1]
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()  # shape: [d_model/2]

        # 使用正弦和余弦函数填充权重
        w[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        w[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦

        # 创建嵌入层并设置固定权重
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        输入:
            x: shape [batch_size, seq_len] 或包含索引的张量
        返回:
            固定嵌入向量: shape [batch_size, seq_len, d_model]
        """
        return self.emb(x).detach()  # 使用detach()确保不计算梯度


class TemporalEmbedding(nn.Module):
    """
    时间嵌入模块：将时间特征（月、日、星期、小时、分钟）嵌入为向量表示
    
    参数:
        d_model: 嵌入维度
        embed_type: 嵌入类型，'fixed'表示使用固定嵌入
        freq: 时间频率，'h'代表小时级，'t'代表分钟级
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        # 根据embed_type选择嵌入类型
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        # 根据频率决定是否包含分钟嵌入
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        输入:
            x: shape [batch_size, seq_len, 5]，时间特征张量，
               格式为[月份, 日期, 星期几, 小时, 分钟]
        返回:
            时间嵌入向量: shape [batch_size, seq_len, d_model]
        """
        x = x.long()  # 将输入转换为长整型
        # 获取各个时间特征的嵌入向量，并相加
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.  # 分钟嵌入
        hour_x = self.hour_embed(x[:, :, 3])  # 小时嵌入
        weekday_x = self.weekday_embed(x[:, :, 2])  # 星期嵌入
        day_x = self.day_embed(x[:, :, 1])  # 日期嵌入
        month_x = self.month_embed(x[:, :, 0])  # 月份嵌入

        # 所有时间特征嵌入相加得到最终的时间嵌入
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    时间特征嵌入模块：将时间特征通过线性层映射为嵌入向量
    
    参数:
        d_model: 嵌入维度
        embed_type: 嵌入类型，通常为'timeF'
        freq: 时间频率，决定输入维度
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # 根据频率确定输入维度
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        # 线性映射层，没有偏置
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """
        输入:
            x: shape [batch_size, seq_len, d_inp] 时间特征
        返回:
            嵌入向量: shape [batch_size, seq_len, d_model]
        """
        return self.embed(x)  # 通过线性层映射


class DataEmbedding(nn.Module):
    """
    数据嵌入模块：结合值嵌入、位置嵌入和时间嵌入
    
    参数:
        c_in: 输入通道数/特征维度
        d_model: 嵌入维度
        embed_type: 时间嵌入类型
        freq: 时间频率
        dropout: Dropout比率
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # 值嵌入：将输入特征映射到嵌入空间
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 位置嵌入：添加位置信息
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 时间嵌入：添加时间信息
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        # Dropout层，用于正则化
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        输入:
            x: shape [batch_size, seq_len, c_in] 输入序列
            x_mark: shape [batch_size, seq_len, num_time_features] 时间特征
        返回:
            嵌入向量: shape [batch_size, seq_len, d_model]
        """
        # 如果没有时间标记，只使用值嵌入和位置嵌入
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x).to(x.device)
        # 否则，结合值嵌入、时间嵌入和位置嵌入
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)  # 应用dropout并返回


class DataEmbedding_wo_pos(nn.Module):
    """
    不带位置嵌入的数据嵌入模块：只结合值嵌入和时间嵌入
    
    参数:
        c_in: 输入通道数/特征维度
        d_model: 嵌入维度
        embed_type: 时间嵌入类型
        freq: 时间频率
        dropout: Dropout比率
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        # 值嵌入：将输入特征映射到嵌入空间
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 位置嵌入：虽然定义了但在前向传播中没有使用
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 时间嵌入：添加时间信息
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        # Dropout层，用于正则化
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        输入:
            x: shape [batch_size, seq_len, c_in] 输入序列
            x_mark: shape [batch_size, seq_len, num_time_features] 时间特征
        返回:
            嵌入向量: shape [batch_size, seq_len, d_model]
        """
        # 如果没有时间标记，只使用值嵌入
        if x_mark is None:
            x = self.value_embedding(x)
        # 否则，结合值嵌入和时间嵌入，但不使用位置嵌入
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)  # 应用dropout并返回


class ReplicationPad1d(nn.Module):
    """
    一维复制填充模块：在序列末尾进行填充，通过复制序列最后一个值
    
    参数:
        padding: 填充配置
    """
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        """
        输入:
            input: shape [batch_size, channels, seq_len] 输入张量
        返回:
            填充后的张量: shape [batch_size, channels, seq_len+padding]
        """
        # 获取最后一个时间步的值并重复padding[-1]次
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        # 将原始输入和填充部分连接起来
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class PatchEmbedding(nn.Module):
    """
    补丁嵌入模块：将输入序列分割成补丁并嵌入
    
    参数:
        d_model: 嵌入维度
        patch_len: 每个补丁的长度
        stride: 补丁之间的步长
        dropout: Dropout比率
    """
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # 补丁化参数
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # 骨干网络，输入编码：将特征向量投影到d维向量空间
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # 位置嵌入（注释掉了，未使用）
        # self.position_embedding = PositionalEmbedding(d_model)

        # 残差dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入:
            x: shape [batch_size, n_vars, seq_len] 输入序列
        返回:
            嵌入向量: shape [batch_size*n_vars, num_patches, d_model]
            n_vars: 变量数量
        """
        # 进行补丁化处理
        n_vars = x.shape[1]  # 获取变量数量
        x = self.padding_patch_layer(x)  # 对序列进行填充，shape: [batch_size, n_vars, seq_len+stride]
        # 将序列划分为补丁，shape: [batch_size, n_vars, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # 重塑张量形状，将批次和变量维度合并，shape: [batch_size*n_vars, num_patches, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # 输入编码
        x = self.value_embedding(x)  # shape: [batch_size*n_vars, num_patches, d_model]
        return self.dropout(x), n_vars  # 返回嵌入向量和变量数量


class DataEmbedding_wo_time(nn.Module):
    """
    不带时间嵌入的数据嵌入模块：只结合值嵌入和位置嵌入
    
    参数:
        c_in: 输入通道数/特征维度
        d_model: 嵌入维度
        embed_type: 嵌入类型（未使用）
        freq: 时间频率（未使用）
        dropout: Dropout比率
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        # 值嵌入：将输入特征映射到嵌入空间
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 位置嵌入：添加位置信息
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # Dropout层，用于正则化
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        输入:
            x: shape [batch_size, seq_len, c_in] 输入序列
        返回:
            嵌入向量: shape [batch_size, seq_len, d_model]
        """
        # 结合值嵌入和位置嵌入，不使用时间嵌入
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)  # 应用dropout并返回
