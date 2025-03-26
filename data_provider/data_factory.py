from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from data_provider.mimic_dataloader import MIMICDataset
from torch.utils.data import DataLoader
import torch

"""
这个函数做的事情就是**"根据配置自动选择合适的数据集类，
构建并返回对应的 Dataset + DataLoader"**。
它是个典型的工具函数，让主程序里只需一行 data_set, data_loader = data_provider(args, 'train') 即可获得想要的批量数据迭代器
"""


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'MIMIC': MIMICDataset,
}

# 自定义的collate函数，用于处理MIMIC数据集中的变长文本数据
def mimic_collate_fn(batch):
    # 解包批次数据
    ts_data, dec_seq, ts_mark, dec_mark, label, text_info = zip(*batch)
    
    # 对于固定长度的张量，直接堆叠
    ts_data = torch.stack(ts_data)
    dec_seq = torch.stack(dec_seq)
    ts_mark = torch.stack(ts_mark)
    dec_mark = torch.stack(dec_mark)
    label = torch.stack(label)
    
    # 对于变长的文本数据，使用列表保存，不进行堆叠
    batch_text_info = {
        'text': [info['text'] for info in text_info],
        'time': [info['time'] for info in text_info]
    }
    
    return ts_data, dec_seq, ts_mark, dec_mark, label, batch_text_info


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    elif args.data == 'MIMIC':
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            task=args.task,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=True,
            test_mode=args.test_mode
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    
    # 对MIMIC数据集使用自定义的collate_fn
    if args.data == 'MIMIC':
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=mimic_collate_fn)
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader
