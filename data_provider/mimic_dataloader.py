import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset

class MIMICDataset(Dataset):
    """
    MIMIC数据集类，用于加载处理好的MIMIC数据，包含时间序列和临床文本数据
    
    参数:
        root_path: 数据根目录
        flag: 数据集类型 ('train', 'val', 'test')
        task: 任务类型 ('ihm', 'pheno', 'los')，分别对应48小时死亡预测，24小时表型预测和住院时长预测
        size: [seq_len, label_len, pred_len]
        features: 特征类型，'M'表示多变量
        target: 目标变量（未使用）这里用的不是csv，而是pkl文件所以用不到
        scale: 是否进行归一化
        test_mode: 是否为测试模式
    """
    def __init__(self, root_path, flag='train', task='ihm', 
                 size=None, features='M', target=None, 
                 scale=True, test_mode=False, **kwargs):
        # 初始化参数
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.task = task  # 'ihm' 或 'pheno'
        self.test_mode = test_mode  # 添加test_mode参数
        
        # 序列长度相关参数
        if size is None:
            self.seq_len = 48  # 对于MIMIC，使用48小时数据
            self.label_len = 0
            self.pred_len = 1  # 对于分类任务，pred_len=1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # 其他参数
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        
        self.__read_data__()
        
    def __read_data__(self):
        """读取MIMIC数据集"""
        # 根据不同的数据集类型选择文件
        if self.flag == 'train':
            data_file = os.path.join(self.root_path, 'train_p2x_data.pkl')
        elif self.flag == 'val':
            data_file = os.path.join(self.root_path, 'val_p2x_data.pkl')
        else:  # test
            data_file = os.path.join(self.root_path, 'test_p2x_data.pkl')
            
        # 加载数据
        with open(data_file, 'rb') as f:
            all_data = pickle.load(f)
            
        # 测试模式下只使用前60条数据
        if self.test_mode:
            self.data = all_data[40:100]
        else:
            self.data = all_data
            
        # 设置样本数量
        self.samples_num = len(self.data)
        
    def __getitem__(self, index):
        """获取单个样本"""
        sample = self.data[index]
        
        # 获取时间序列数据 - 使用reg_ts字段（规则化后的数据）
        # reg_ts形状为[48, 34]，前17列是特征，后17列是掩码
        ts_data = sample['reg_ts'][:, :17]  # 只使用前17个特征
        
        # 获取文本数据和对应的时间戳
        text_data = sample['text_data']
        text_time = sample['text_time']
        
        # 获取标签
        if self.task == 'ihm':
            # 48小时院内死亡预测 - 二分类任务
            label = sample['label']  # 假设第一个元素是死亡标签(0或1)
        elif self.task == 'pheno':
            # 24小时表型预测 - 多标签分类任务
            label = sample['label']  # 假设是25个病症的多标签向量
        else:  # 'los'
            # 住院时长预测（未实现）
            label = sample['label'][0]  # 临时使用，实际可能需要转换为适当的值
        
        # 将数据转换为tensor
        ts_data = torch.FloatTensor(ts_data)
        label = torch.LongTensor([label])
        
        # 如果是预测任务，我们需要dec_seq，这里简化处理
        # 创建空的decoder输入序列
        dec_seq = torch.zeros((self.seq_len, 17)).float()
        
        # 创建空的时间特征（原模型使用但我们暂不需要）
        ts_mark = torch.zeros((self.seq_len, 4)).float()
        dec_mark = torch.zeros((self.seq_len, 4)).float()
        
        # 将文本数据和时间戳打包在一起返回
        text_info = {
            'text': text_data[-5:],
            'time': text_time[-5:]
        }
        
        return ts_data, dec_seq, ts_mark, dec_mark, label, text_info
    
    def __len__(self):
        """返回数据集大小"""
        return self.samples_num 
    

if __name__ == '__main__':
    # 这里的 root_path 请替换成你真实的MIMIC数据目录
    root_path = '/home/ubuntu/Virginia/output_mimic3/ihm'
    
    # 实例化一个训练集示例
    dataset = MIMICDataset(root_path=root_path, flag='train', task='ihm')
    print("Dataset size:", len(dataset))
    
    # 获取并打印第一个样本
    sample_index = 0
    ts_data, dec_seq, ts_mark, dec_mark, label, text_info = dataset[sample_index]
    
    print("ts_data shape:", ts_data)   # [48, 17]
    print("dec_seq shape:", dec_seq.shape)   # [48, 17]
    print("ts_mark shape:", ts_mark.shape)   # [48, 4]
    print("dec_mark shape:", dec_mark.shape) # [48, 4]
    print("label shape:", label.shape)       # [1]，二分类就是一个值
    
    print("label (content):", label.item())  # 若为二分类，可打印出0或1
    print("text_info:", text_info)           # 包含 'text' 和 'time' 两部分
    print("text_data_length:", len(text_info['text']))  # 打印文本数据
