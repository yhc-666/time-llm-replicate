from math import sqrt
import numpy as np
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
P: patch数量 (patch_nums)
d_model: Patch embedding 的维度
d_llm: LLM embedding 维度
d_ff: d_ff fc layer 维度
"""


class FlattenHead(nn.Module):
    """
    通道注意力特征投影头, 接受llm输出做最终预测
    """
    def __init__(self, n_vars, nf, output_size, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Linear(nf, output_size)
        self.dropout = nn.Dropout(head_dropout)
        self.sigmoid = nn.Sigmoid()
        
        # 通道注意力网络
        self.channel_attn = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Linear(output_size, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 输入x形状: [B, N, d_ff, patch_num]
        x = self.flatten(x)  # [B, N, d_ff*patch_num]
        x = self.linear(x)   # [B, N, output_size]
        
        # 计算通道权重 [B, N, 1]
        weights = self.channel_attn(x)
        
        # 加权聚合通道 [B, output_size]
        x = (x * weights).sum(dim=1)
        
        x = self.dropout(x)
        x = self.sigmoid(x)
        return x


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


class Model(nn.Module):
    """
    MIMIC-TimeLLM模型: 针对MIMIC-III数据集的TimeLLM变种，支持结合文本笔记的时间序列预测
    
    扩展了原TimeLLM模型，加入了:
    1. 文本处理组件，能够处理临床笔记
    2. 为IHM和PHE任务优化的分类输出头
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.task = configs.task  # 'ihm', 'pheno', 或 'los'

        # 1) 加载和配置预训练语言模型(LLAMA、GPT2或BERT)
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
            self.d_llm = self.llama_config.hidden_size
        elif configs.llm_model == 'GPT2':
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
            self.d_llm = self.gpt2_config.n_embd
        elif configs.llm_model == 'BERT':
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
            self.d_llm = self.bert_config.hidden_size
        elif configs.llm_model == 'BERT_TINY':
            model_name = "prajjwal1/bert-tiny"
            # 也可换成 "prajjwal1/bert-mini" 等其它更小的BERT变体

            self.bert_config = BertConfig.from_pretrained(model_name)
            # 下面两行仅在需要覆盖默认配置时启用(例如修改 num_hidden_layers)：
            # self.bert_config.num_hidden_layers = configs.llm_layers
            # self.bert_config.output_attentions = True
            # self.bert_config.output_hidden_states = True

            try:
                self.llm_model = BertModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local BERT-Tiny model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local BERT-Tiny tokenizer files not found. Attempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=False
                )

            self.d_llm = self.bert_config.hidden_size

        else:
            raise Exception('LLM model is not defined')
        
        self.llm_model.gradient_checkpointing_enable()

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
        self.description = "MIMIC indicates real-world in-hospital patient records. Each patient's record consists of 48 time steps (hourly) of multivariate vital signals, plus clinical text notes. We focus on In-Hospital Mortality (IHM) prediction."

        self.dropout = nn.Dropout(configs.dropout)

        # 2) 时间序列补丁嵌入层
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        # 3) 词嵌入和映射
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        # 将完整词表嵌入(通常有几万个词)映射到更小的"文本原型"空间(1000维)
        # 这一步很关键，通过降维创建了更紧凑的文本表示，作为时间序列跨模态对齐的目标
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # 4) 重编程层 - 核心技术
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # 新增: 特征投影MLP
        self.feature_projection = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm // 2),
            nn.ReLU(),
            nn.Linear(self.d_llm // 2, self.d_ff)
        )

        # 5) 计算输出投影参数
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # 根据任务类型确定输出层
        if self.task == 'ihm':
            # 48小时院内死亡预测 - 二分类任务
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, 1, 
                                                head_dropout=configs.dropout)
        elif self.task == 'pheno':
            # 24小时表型预测 - 25分类多标签任务
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, 25, 
                                                head_dropout=configs.dropout)
        elif self.task == 'los':
            # 住院时长预测 - 回归任务
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, 1, 
                                                head_dropout=configs.dropout)
        else:
            raise NotImplementedError(f"Task {self.task} not implemented")

        # 在__init__方法中添加MLP投影层
        self.hidden_projection = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm // 2),
            nn.ReLU(),
            nn.Linear(self.d_llm // 2, self.d_ff)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_info=None):
        """
        模型前向传播
        
        参数:
            x_enc: 输入时间序列 [B, T, N]
            x_mark_enc: 输入时间特征标记 [B, T, 4]
            x_dec: 解码器输入 [B, T, N]
            x_mark_dec: 解码器时间特征标记 [B, T, 4]
            text_info: 包含文本数据和对应时间戳的字典 {'text': list, 'time': list}
        
        返回:
            dec_out: 任务输出
                - 对于IHM: [B, 1] - 每个样本的死亡概率
                - 对于PHE: [B, 25] - 每个样本25个表型的概率
                - 对于LOS: [B, 1] - 每个样本的预测住院时长
        """
        # 数据准备
        B, T, N = x_enc.shape  # B: 批大小, T: 序列长度, N: 变量数
        
        # 将形状从[B, T, N]转换为[B*N, T, 1]，便于计算每个变量的统计特征
        x_enc_reshaped = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # 计算统计特征，用于生成提示词
        min_values = torch.min(x_enc_reshaped, dim=1)[0]     # 最小值，形状: [B*N, 1]
        max_values = torch.max(x_enc_reshaped, dim=1)[0]     # 最大值，形状: [B*N, 1]
        medians = torch.median(x_enc_reshaped, dim=1).values # 中位数，形状: [B*N, 1]
        lags = self.calcute_lags(x_enc_reshaped)             # 自相关lag，形状: [B*N, top_k]
        trends = x_enc_reshaped.diff(dim=1).sum(dim=1)       # 趋势(上升/下降)，形状: [B*N, 1]
        
        # 为每个样本和变量生成提示词
        prompt = []
        for b_n in range(x_enc_reshaped.shape[0]):
            b = b_n // N  # 样本索引
            n = b_n % N   # 变量索引
            
            min_val = f"{min_values[b_n].tolist()[0]:.2f}"
            max_val = f"{max_values[b_n].tolist()[0]:.2f}"
            median_val = f"{medians[b_n].tolist()[0]:.2f}"
            trend = "upward" if trends[b_n].item() > 0 else "downward" if trends[b_n].item() < 0 else "stable"
            lag_vals = ", ".join([f"{lag}" for lag in lags[b_n]])
            
            # 从text_info获取文本信息
            text_notes = ""
            if text_info is not None:
                sample_texts = text_info['text'][b]  # 当前样本的所有文本
                sample_times = text_info['time'][b]  # 对应的时间戳
                
                # 限制最多显示5条文本记录以防止超出token限制
                max_notes = 5
                if len(sample_texts) > 0:
                    for i in range(min(len(sample_texts), max_notes)):
                        hour = sample_times[i]
                        note = sample_texts[i]
                        # 限制笔记长度
                        if len(note) > 100:
                            note = note[:100] + "..."
                        text_notes += f"(time={hour:.1f}h) \"{note}\"\n"
                else:
                    text_notes = "(No clinical notes available)"
            else:
                text_notes = "(No clinical notes available)"
            
            # 构建提示词 - 医疗场景特有的提示词格式
            task_instruction = ""
            if self.task == 'ihm':
                task_instruction = "Predict whether this patient will die (IHM) given:"
            elif self.task == 'pheno':
                task_instruction = "Predict which of 25 phenotypes this patient has given:"
            elif self.task == 'los':
                task_instruction = "Predict remaining length of stay for this patient given:"
            
            prompt_ = f"""[BEGIN DATA]
***
[Domain]:
We are currently focusing on vital-signal Feature {n}.
We sample this specific feature every hour, for {self.seq_len} time steps in total.
Clinical text notes are recorded at certain hours as well.

***
[Instruction]:
{task_instruction}
1) Structural time-series data (for Feature {n}).
2) Clinical text notes at certain times.

***
[Statistics]:
- Minimum: {min_val}
- Maximum: {max_val}
- Median: {median_val}
- Overall trend: {trend}
- Top five lags: {lag_vals}

***
[Text Notes with Timestamps]:
We have M notes, each recorded at a specific hour:
{text_notes}

***
[END DATA]
"""
            prompt.append(prompt_)

        
        # 处理提示词 - 将文本转换为token IDs和嵌入，与TimeLLM保持一致
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(x_enc.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt)
        
        # 构建文本原型
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        
        # --- 以下是核心处理流程，保持与TimeLLM一致 ---
        
        # 时间序列补丁嵌入和重编程
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # 形状: [B, N, T]
        # PatchEmbedding将时间序列切分成patch并嵌入到d_model维度空间
        enc_out, n_vars = self.patch_embedding(x_enc)
        #enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # 通过重编程层将时间序列表示与LLM文本空间对齐
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        # enc_out: [544, 48, 768]
        
        # 连接提示词嵌入和重编程后的时间序列嵌入
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # llama_enc_out: [544, 447, 768]
        
        # 通过LLM提取特征
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        
        # 使用MLP投影将特征从d_llm维度映射到d_ff维度
        batch_size, seq_len, _ = dec_out.shape
        dec_out_flat = dec_out.reshape(-1, self.d_llm)
        dec_out_flat = self.hidden_projection(dec_out_flat)
        dec_out = dec_out_flat.reshape(batch_size, seq_len, self.d_ff)
        
        # 重塑输出以适应FlattenHead
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # 维度调整：[B*N, seq_len, d_ff] -> [B, N, seq_len, d_ff]
        
        # 交换最后两个维度，使特征维度在第三位置
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        # 形状变为: [B, N, d_ff, seq_len]
        
        # 生成最终预测，只使用最后patch_nums个位置
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        
        return dec_out

    def calcute_lags(self, x_enc):
        """
        计算时间序列的自相关最强的lag值
        
        参数:
            x_enc: 形状为[B, T, 1]的输入时间序列
            
        返回:
            top-k lag值，形状为[B, top_k]
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
