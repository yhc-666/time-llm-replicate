import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import wandb  # 导入wandb库

from models import MIMIC_TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

"""
python run_mimic.py --use_wandb --wandb_project "MIMIC-TimeLLM" --wandb_entity "您的用户名"
"""



# 环境设置
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, load_content

parser = argparse.ArgumentParser(description='MIMIC-TimeLLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 基础配置
parser.add_argument('--task_name', type=str, default='classification',
                    help='task name, for MIMIC we use classification')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='mimic', help='model id')
parser.add_argument('--model_comment', type=str, default='mimic_ihm', help='prefix when saving test results')
parser.add_argument('--model', type=str, default='MIMIC_TimeLLM',
                    help='model name, only MIMIC_TimeLLM is supported')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# 数据加载器
parser.add_argument('--data', type=str, default='MIMIC', help='dataset type')
parser.add_argument('--root_path', type=str, default='/Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/output_mimic3/ihm', help='root path of the data file')
parser.add_argument('--task', type=str, default='ihm', help='task type, options: [ihm, pheno, los]')
parser.add_argument('--features', type=str, default='M', help='forecasting task, M:multivariate')
parser.add_argument('--target', type=str, default=None, help='target feature')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 序列长度设置
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

# 模型定义
parser.add_argument('--enc_in', type=int, default=17, help='encoder input size, 17 for MIMIC')
parser.add_argument('--dec_in', type=int, default=17, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=2, help='patch length')
parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--prompt_domain', type=int, default=1, help='whether to use domain prompt')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')

# 优化设置
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='BCE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
# wandb配置参数
parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging', default=False)
parser.add_argument('--wandb_project', type=str, default='MIMIC-TimeLLM', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
# 添加控制分布式训练的参数
parser.add_argument('--disable_distributed', action='store_true', help='禁用分布式训练', default=False)
# mac测试模式
parser.add_argument('--test_mode', action='store_true', help='快速测试模式，仅使用300条数据', default=False)

args = parser.parse_args()

# 根据参数决定是否使用分布式训练
if args.disable_distributed:
    # 禁用分布式训练，使用单进程模式
    accelerator = Accelerator(cpu=True, mixed_precision=None)
else:
    # 使用分布式训练
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # 修改DeepSpeed配置
    if args.use_amp:
        mixed_precision = "bf16"
    else:
        mixed_precision = None
    # 使用accelerator处理mixed_precision而不是通过DeepSpeed插件
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs], 
        mixed_precision=mixed_precision,
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    )

for ii in range(args.itr):
    # 设置实验记录
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.task, ii)
    
    # 初始化wandb（只在主进程上）
    if args.use_wandb and accelerator.is_local_main_process:
        run_name = f"{setting}-{args.model_comment}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args)  # 记录所有参数
        )
        # 记录模型配置
        wandb.config.update({
            "setting": setting,
            "model_comment": args.model_comment
        })

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # 创建模型
    model = MIMIC_TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    # 只训练需要梯度的参数
    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    # 学习率调度器
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)

    # 根据任务类型选择不同的损失函数
    if args.task == 'ihm' or args.task == 'pheno':
        criterion = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss替代BCELoss
    if args.task == 'los':
        criterion = nn.MSELoss()  # 连续值预测

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    # 模型训练函数
    def train_epoch(epoch):
        model.train()
        total_loss = []
        epoch_time = time.time()
        iter_count = 0
        time_now = time.time()  # 初始化time_now变量
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_label, text_info) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            batch_label = batch_label.float().to(accelerator.device)

            # 前向传播
            if args.use_amp:
                with accelerator.autocast():
                    outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark, text_info)
                    outputs = outputs.to(torch.float32)
                    loss = criterion(outputs, batch_label)
            else:
                outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark, text_info)
                outputs = outputs.to(torch.float32)
                loss = criterion(outputs, batch_label)
                
            total_loss.append(loss.item())

            if (i + 1) % 50 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                
                # 记录到wandb
                if args.use_wandb and accelerator.is_local_main_process:
                    wandb.log({
                        "iter_loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0] if args.lradj == 'TST' else model_optim.param_groups[0]['lr'],
                        "epoch": epoch + (i + 1) / train_steps,
                    })
                
                iter_count = 0
                time_now = time.time()

            # 反向传播
            accelerator.backward(loss)
            model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(total_loss)
        
        # 记录每个epoch的训练损失
        if args.use_wandb and accelerator.is_local_main_process:
            wandb.log({
                "train_loss": train_loss,
                "epoch": epoch + 1
            })
            
        return train_loss

    # 验证函数 - 计算AUROC和AUPRC
    def validate(data_loader):
        model.eval()
        total_loss = []
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_label, text_info) in enumerate(data_loader):
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)
                batch_label = batch_label.float().to(accelerator.device)

                # 前向传播
                outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark, text_info)
                outputs = outputs.to(torch.float32)
                
                # 计算损失
                loss = criterion(outputs, batch_label)
                total_loss.append(loss.item())
                    
                # 对于评估指标，使用sigmoid将logits转换为概率
                probs = torch.sigmoid(outputs)
                
                # 收集预测和真实值
                preds.append(probs.detach().cpu().numpy())
                trues.append(batch_label.detach().cpu().numpy())

        # 合并批次结果
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 计算评估指标
        if args.task == 'ihm':
            # 二分类: 计算 AUROC, AUPRC, F1, 准确率
            auroc = roc_auc_score(trues, preds)
            auprc = average_precision_score(trues, preds)
            
            # 将概率转换为二分类标签 (0.5阈值)
            binary_preds = (preds > 0.5).astype(int)
            f1 = f1_score(trues, binary_preds)
            acc = accuracy_score(trues, binary_preds)
            
            eval_result = {
                'loss': np.average(total_loss),
                'auroc': auroc,
                'auprc': auprc,
                'f1': f1,
                'accuracy': acc
            }
        elif args.task == 'pheno':
            # 多标签分类: 计算每个标签的 AUROC 和 AUPRC，然后求平均
            all_auroc = []
            all_auprc = []
            for i in range(preds.shape[1]):
                if len(np.unique(trues[:, i])) > 1:  # 只有当有正负样本时才计算
                    all_auroc.append(roc_auc_score(trues[:, i], preds[:, i]))
                    all_auprc.append(average_precision_score(trues[:, i], preds[:, i]))
            
            eval_result = {
                'loss': np.average(total_loss),
                'auroc': np.mean(all_auroc) if all_auroc else 0,
                'auprc': np.mean(all_auprc) if all_auprc else 0
            }
        else:
            # 回归任务: 使用MSE作为评估指标
            eval_result = {
                'loss': np.average(total_loss)
            }
            
        return eval_result

    # 主训练循环
    for epoch in range(args.train_epochs):
        train_loss = train_epoch(epoch)
        
        # 验证
        eval_results = validate(vali_loader)
        test_results = validate(test_loader)
        
        # 打印结果
        if args.task == 'ihm':
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Val Loss: {2:.7f} Val AUROC: {3:.4f} Val AUPRC: {4:.4f} "
                "Test Loss: {5:.7f} Test AUROC: {6:.4f} Test AUPRC: {7:.4f}".format(
                    epoch + 1, train_loss, 
                    eval_results['loss'], eval_results['auroc'], eval_results['auprc'],
                    test_results['loss'], test_results['auroc'], test_results['auprc']))
            
            # 记录指标到wandb
            if args.use_wandb and accelerator.is_local_main_process:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": eval_results['loss'],
                    "val/auroc": eval_results['auroc'],
                    "val/auprc": eval_results['auprc'],
                    "val/f1": eval_results['f1'],
                    "val/accuracy": eval_results['accuracy'],
                    "test/loss": test_results['loss'],
                    "test/auroc": test_results['auroc'],
                    "test/auprc": test_results['auprc'],
                    "test/f1": test_results['f1'],
                    "test/accuracy": test_results['accuracy']
                })
        elif args.task == 'pheno':
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Val Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_loss, eval_results['loss'], test_results['loss']))
            
            # 记录指标到wandb
            if args.use_wandb and accelerator.is_local_main_process:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": eval_results['loss'],
                    "val/auroc": eval_results.get('auroc', 0),
                    "val/auprc": eval_results.get('auprc', 0),
                    "test/loss": test_results['loss'],
                    "test/auroc": test_results.get('auroc', 0),
                    "test/auprc": test_results.get('auprc', 0)
                })
        else:
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Val Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_loss, eval_results['loss'], test_results['loss']))
            
            # 记录指标到wandb
            if args.use_wandb and accelerator.is_local_main_process:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": eval_results['loss'],
                    "test/loss": test_results['loss']
                })

        # 早停
        early_stopping(eval_results['loss'], model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        # 调整学习率
        if args.lradj != 'TST':
            adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    # 在测试集上进行最终评估
    model.load_state_dict(torch.load(os.path.join(path, 'checkpoint')))
    final_test_results = validate(test_loader)
    
    if args.task == 'ihm':
        accelerator.print(
            "Final Test Results: Loss: {0:.7f} AUROC: {1:.4f} AUPRC: {2:.4f} F1: {3:.4f} Accuracy: {4:.4f}".format(
                final_test_results['loss'], final_test_results['auroc'], 
                final_test_results['auprc'], final_test_results['f1'], 
                final_test_results['accuracy']))
                
        # 记录最终测试结果
        if args.use_wandb and accelerator.is_local_main_process:
            wandb.run.summary.update({
                "final_test/loss": final_test_results['loss'],
                "final_test/auroc": final_test_results['auroc'],
                "final_test/auprc": final_test_results['auprc'],
                "final_test/f1": final_test_results['f1'],
                "final_test/accuracy": final_test_results['accuracy']
            })
    elif args.task == 'pheno':
        accelerator.print(
            "Final Test Results: Loss: {0:.7f} AUROC: {1:.4f} AUPRC: {2:.4f}".format(
                final_test_results['loss'], final_test_results['auroc'], 
                final_test_results['auprc']))
                
        # 记录最终测试结果
        if args.use_wandb and accelerator.is_local_main_process:
            wandb.run.summary.update({
                "final_test/loss": final_test_results['loss'],
                "final_test/auroc": final_test_results['auroc'],
                "final_test/auprc": final_test_results['auprc']
            })
    else:
        accelerator.print(
            "Final Test Results: Loss: {0:.7f}".format(final_test_results['loss']))
            
        # 记录最终测试结果
        if args.use_wandb and accelerator.is_local_main_process:
            wandb.run.summary.update({
                "final_test/loss": final_test_results['loss']
            })
            
    # 结束wandb会话
    if args.use_wandb and accelerator.is_local_main_process:
        wandb.finish()

accelerator.wait_for_everyone()
accelerator.print('Training complete!') 