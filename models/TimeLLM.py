from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

# 设置transformers库的日志级别为ERROR，以减少不必要的日志输出
transformers.logging.set_verbosity_error()


"""
维度符号说明:
B: 批大小 (batch size)
T: 单条序列的时间步数 (sequence length)
N: 通道数/变量数 (number of variables)
H: 要预测的步数 (prediction horizon)
P: patch数量 (patch_nums)   \ 每个patch最终对应一个输入 llm的embedding token
d_model: Patch embedding 的维度
d_llm: LLM hidden state 维度
d_ff: d_ff被用作cross attn 机制中每个头的查询维度大小
"""



"""
关于timellm的mask
关于padding mask - 通常用于处理批处理中不同长度序列的填充部分。尽管代码中对prompt使用了padding：
Apply to TimeLLM.py
input_ids
但没有显式提取和使用attention_mask，这可能是因为：
时间序列输入长度固定
提示词部分没有考虑padding的影响
关于causal mask - 确实不需要考虑，因为：
TimeLLM使用LLM作为特征提取器，而非生成模型
时间序列预测任务中，我们希望模型能看到整个历史序列的全部信息
在特征提取阶段，双向注意力机制更有利于捕获时间序列的完整模式
在这种设计下，让每个token都能关注所有其他token是合理的，有助于更全面地捕获序列信息。
"""


class FlattenHead(nn.Module):
    """
    将高维特征展平并投影为预测序列的输出头部
    
    参数:
        n_vars (int): 变量数量
        nf (int): 输入特征维度 (F * P)
        target_window (int): 目标预测窗口大小
        head_dropout (float): dropout率
        
    输入: 
        x: 形状为 [B, N, F, P]
            B: batch size
            N: 变量数量
            F: 特征维度 (通常是d_ff)
            P: patch 数量
            
    输出:
        形状为 [B, N, H]: 每个批次的每个变量的H步预测
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        # 从索引-2开始展平，把最后两个维度(F,P)合并为一个维度(F*P)
        self.flatten = nn.Flatten(start_dim=-2)
        # 将展平的特征映射到目标预测长度
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # 输入x形状: [B, N, F, P]
        x = self.flatten(x)  # 形状变为: [B, N, F*P]
        x = self.linear(x)   # 形状变为: [B, N, H]
        x = self.dropout(x)  # 形状不变: [B, N, H]
        return x


class Model(nn.Module):
    """
    TimeLLM模型: 利用大型语言模型进行时间序列预测的框架
    
    核心思想:
    1. 将时间序列切分成小片段(patches)，并嵌入到一个向量空间
    2. 使用一种跨模态重编程技术，将这些时间序列表示与LLM的文本嵌入对齐
    3. 使用自然语言提示词增强模型对时间序列的理解
    4. 利用LLM进行特征提取，不需要微调LLM本身
    """
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len  # 预测长度H
        self.seq_len = configs.seq_len    # 输入序列长度T
        self.d_ff = configs.d_ff          # 查询维度大小(d_keys)
        self.top_k = 5                    # 计算自相关时选择的顶部lag数
        self.d_llm = configs.llm_dim      # LLM的隐藏状态维度
        self.patch_len = configs.patch_len # 切片长度
        self.stride = configs.stride      # 切片步长

        # 1) 加载和配置预训练语言模型(LLAMA、GPT2或BERT)
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            # 自定义LLM层数，启用注意力和隐藏状态输出
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                # 先尝试从本地加载模型
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                # 本地模型不存在，从HuggingFace下载
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                # 尝试从本地加载tokenizer
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                # 本地tokenizer不存在，从HuggingFace下载
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            # GPT2模型配置与加载，逻辑与LLAMA类似
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            # BERT模型配置与加载，逻辑与LLAMA和GPT2类似
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        # 确保tokenizer有pad_token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # 冻结LLM参数 - 这是关键步骤，确保只训练少量参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 设置数据集描述，用于生成提示词
        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        # 2) 时间序列补丁嵌入层
        # 将原始时间序列分割成补丁并嵌入到d_model维度空间
        # PatchEmbedding作用:
        # - 对时间序列进行滑动窗口切分，生成固定长度的patch
        # - 将每个patch通过TokenEmbedding投影到d_model维度空间
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        # 3) 词嵌入和映射
        # 获取LLM的词嵌入权重
        self.word_embeddings = self.llm_model.get_input_embeddings().weight  # 形状: [vocab_size, d_llm]
        self.vocab_size = self.word_embeddings.shape[0]

        # 将完整词表嵌入(通常有几万个词)映射到更小的"文本原型"空间(1000维)
        # 这一步很关键，通过降维创建了更紧凑的文本表示，作为时间序列跨模态对齐的目标
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # 4) 重编程层 - 核心技术
        # 通过交叉注意力机制，将时间序列patch表示与LLM的词嵌入空间对齐
        # 这是TimeLLM的核心创新: 使用文本原型作为目标空间，实现跨模态对齐
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # 5) 计算输出投影参数
        # 计算patch数量并设置head_nf(特征总维度)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # 输出投影层 - 将LLM特征转换为最终预测结果
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        # 6) 归一化层 - 处理输入和输出
        # Normalize类同时负责输入归一化和输出反归一化
        # 确保模型输入数据规模适当，且输出恢复到原始数据尺度
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        模型前向传播
        
        参数:
            x_enc: 形状为[B, T, N]的输入时间序列
            x_mark_enc: 输入序列的时间特征(可能未使用)
            x_dec: 用于解码的输入(可能未使用)
            x_mark_dec: 解码输入的时间特征(可能未使用)
            mask: 可选的掩码(可能未使用)
            
        返回:
            预测结果，形状为[B, pred_len, N]
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 只返回预测长度的部分
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        时间序列预测的核心函数
        
        参数:
            x_enc: 形状为[B, T, N]的输入时间序列
            x_mark_enc: 输入序列的时间特征(未使用)
            x_dec: 用于解码的输入(未使用)
            x_mark_dec: 解码输入的时间特征(未使用)
            
        返回:
            预测结果，形状为[B, pred_len, N]
        
        处理流程:
        1. 输入归一化
        2. 计算时间序列统计特征(最小值、最大值、中位数、趋势、自相关lag)
        3. 基于统计特征生成自然语言提示词
        4. 时间序列切片和嵌入
        5. 将时间序列表示通过重编程对齐到LLM空间
        6. 通过LLM提取特征
        7. 将LLM输出投影到预测空间
        8. 反归一化得到最终预测
        """
        # 1. 输入归一化
        x_enc = self.normalize_layers(x_enc, 'norm')  # 形状: [B, T, N]

        # 2. 重塑输入以便于计算统计量
        B, T, N = x_enc.size()  # B:批大小, T:序列长度, N:变量数
        # 将形状从[B, T, N]转换为[B*N, T, 1]
        # 这一步将多变量时序数据转换为单变量时序的集合，便于计算每个变量的统计特征
        # 这种处理方式使得模型能够为每个变量生成定制化的提示词，包含该变量特有的统计特征，从而增强LLM对不同变量特性的理解，提高预测精度。
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)  # 形状: [B*N, T, 1]

        # 3. 计算时间序列统计特征，用于生成提示词
        min_values = torch.min(x_enc, dim=1)[0]     # 最小值，形状: [B*N, 1]
        max_values = torch.max(x_enc, dim=1)[0]     # 最大值，形状: [B*N, 1]
        medians = torch.median(x_enc, dim=1).values # 中位数，形状: [B*N, 1]
        lags = self.calcute_lags(x_enc)             # 自相关lag，形状: [B*N, top_k]
        trends = x_enc.diff(dim=1).sum(dim=1)       # 趋势(上升/下降)，形状: [B*N, 1]

        # 4. 为每个时间序列生成自然语言提示词
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            # 构建带有时间序列信息的提示词 - 这是"Prompt-as-Prefix"技术的体现
            # 提示词包含: 数据集描述、任务描述、输入统计信息(最小值、最大值、中位数、趋势、顶部lags)
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        # 5. 将输入重塑回原始形状
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()  # 形状: [B, T, N]

        # 6. 处理提示词 - 将文本转换为token IDs和嵌入
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        # 形状: [B*N, prompt_len, d_llm]
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        # 7. 构建文本原型(text prototypes)
        # 将词嵌入矩阵映射到较小的表示空间: [vocab_size, d_llm] -> [num_tokens, d_llm]
        # 这些文本原型是跨模态对齐的关键，作为时间序列的"目标空间"
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # 8. 时间序列补丁嵌入和重编程
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # 形状: [B, N, T]
        # PatchEmbedding将时间序列切分成patch并嵌入到d_model维度空间，生成patch嵌入，形状: [B*N, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # 通过重编程层将时间序列表示与LLM文本空间对齐，形状: [B*N, patch_num, d_llm]
        # 这步是模型的核心: 时间序列跨模态迁移到文本空间
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # 9. 连接提示词嵌入和重编程后的时间序列嵌入
        # 形状: [B*N, prompt_len+patch_num, d_llm]
        # 这实现了"Prompt-as-Prefix"技术，提示词作为上下文前缀
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        # 10. 通过LLM提取特征
        # 形状: [B*N, prompt_len+patch_num, d_llm]
        # 冻结的LLM模型作为特征提取器，将跨模态对齐的时间序列信息转换为富含语义的表示
        """
        last_hidden_state 返回的是模型最后一层Transformer层的输出隐藏状态(hidden states)，即：
        对输入序列中每个token进行编码后的特征表示(embeddings)。
        更具体地：
        Shape 为 (batch_size, sequence_length, hidden_size)。
        包含了输入序列的上下文信息，通常用作后续任务(如分类、回归或生成)所需的特征表示。
        对于 Decoder-only 模型(如 GPT、LLaMA)，它代表模型生成或预测下一个 token 时，每个位置对应的隐藏状态。
        对于 Encoder-only 模型(如 BERT)，它代表输入序列的整体上下文信息，每个位置对应各个 token 编码后的表示。
        每个位置 (batch_i, token_j, :) 代表模型中第j个 token 的上下文感知表示
        """
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # 只保留前d_ff维的特征，形状: [B*N, prompt_len+patch_num, d_ff]
        # ？？？？？这样直接截取是不是不合理？感觉应该用一个线性层投影到d_ff维更合理
        dec_out = dec_out[:, :, :self.d_ff]

        # 11. 重塑输出以适应FlattenHead
        # 形状: [B, N, prompt_len+patch_num, d_ff]
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # 形状: [B, N, d_ff, prompt_len+patch_num]
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # 12. 生成最终预测
        # 只使用最后patch_nums个位置的输出，形状: [B, N, H]
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        # 形状: [B, H, N]
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # 13. 反归一化输出
        # 将归一化的预测结果转换回原始数据尺度
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        """
        计算时间序列的自相关最强的lag值
        
        参数:
            x_enc: 形状为[B, T, 1]的输入时间序列
            
        返回:
            top-k lag值，形状为[B, top_k]
            
        原理:
        使用傅里叶变换计算自相关，然后找出最显著的lag值。
        这些lag值提供了时间序列周期性和重复模式的信息，对预测很有价值。
        """
        # 计算输入的傅里叶变换，形状: [B, 1, T//2+1]
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        # 计算自相关的傅里叶表示(频域乘法等于时域卷积)
        res = q_fft * torch.conj(k_fft)
        # 逆傅里叶变换得到自相关序列，形状: [B, 1, T]
        corr = torch.fft.irfft(res, dim=-1)
        # 计算平均值，形状: [B, T]
        mean_value = torch.mean(corr, dim=1)
        # 找出自相关最强的top_k个lag，形状: [B, top_k]
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    """
    重编程层 - 实现时间序列表示与LLM文本空间的对齐
    
    通过多头交叉注意力机制，使时间序列片段查询(query)文本原型(作为key和value)
    这是TimeLLM最核心的创新之一，实现了时间序列到文本空间的有效迁移，无需微调LLM。
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        """
        参数:
            d_model: 时间序列patch嵌入维度
            n_heads: 注意力头数
            d_keys: 每个头的查询维度 (默认为d_model//n_heads)
            d_llm: LLM的隐藏状态维度
            attention_dropout: 注意力dropout率
        """
        super(ReprogrammingLayer, self).__init__()

        # 如果未指定d_keys，则使用d_model除以头数
        d_keys = d_keys or (d_model // n_heads)

        # 查询投影: 将时间序列表示映射到查询空间
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        # 键投影: 将文本原型映射到键空间
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        # 值投影: 将文本原型映射到值空间
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        # 输出投影: 将注意力输出映射回LLM空间
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        前向传播
        
        参数:
            target_embedding: 时间序列patch嵌入，形状[B, L, d_model]
            source_embedding: 文本原型嵌入，形状[S, d_llm]
            value_embedding: 文本原型嵌入，形状[S, d_llm]
            
        返回:
            重编程后的表示，形状[B, L, d_llm]
        """
        B, L, _ = target_embedding.shape  # B:批大小, L:patch数量
        S, _ = source_embedding.shape     # S:文本原型数量
        H = self.n_heads                  # H:注意力头数

        # 投影并重塑为多头格式
        # 形状: [B, L, H, d_keys]
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        # 形状: [S, H, d_keys]
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        # 形状: [S, H, d_keys]
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # 执行重编程操作(多头注意力)
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        # 重塑输出并进行最终投影
        # 形状: [B, L, H*d_keys]
        out = out.reshape(B, L, -1)
        # 形状: [B, L, d_llm]
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """
        实际的重编程计算(多头交叉注意力)
        
        参数:
            target_embedding: 时间序列查询，形状[B, L, H, E]
            source_embedding: 文本原型键，形状[S, H, E]
            value_embedding: 文本原型值，形状[S, H, E]
            
        返回:
            注意力输出，形状[B, L, H, E]
            
        这是跨模态重编程的核心计算: 时间序列查询与文本原型的交互
        通过注意力机制，时间序列的表示被"重编程"到与文本对齐的空间中
        """
        B, L, H, E = target_embedding.shape  # B:批大小, L:patch数量, H:头数, E:每头维度

        # 注意力缩放因子
        scale = 1. / sqrt(E)

        # 计算注意力分数: query与key的点积，形状: [B, H, L, S]
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        # 应用softmax和dropout得到注意力权重，形状: [B, H, L, S]
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # 加权聚合value，形状: [B, L, H, E]
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
