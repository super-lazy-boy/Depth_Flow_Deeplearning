from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
# import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from model.raft import RAFT
import evaluate
from datasets import datasets

from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace

import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.split_dataset import mk_file
# from utils_training_plot import train_with_plot, plot_curves

try:
    from torch.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 100


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model,train_loader):
    """ Create the optimizer and learning rate scheduler """
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.num_steps

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=total_steps,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def plot_curves(train_losses, train_epe_list, save_path="result"):
    """
    train_losses: 每个 epoch 的平均 loss
    train_epe_list: 每个 epoch 的平均 EPE（我们用它当作“精度指标”的反向度量，越低越好）
    """
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(10,5))
    save_dir = f"{save_path}/{args.name}"
    if save_dir != "" and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loss 曲线
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # EPE 曲线（替代 accuracy）
    plt.subplot(1,2,2)
    plt.plot(epochs, train_epe_list, color='orange', marker='o')
    plt.title('Training EPE')
    plt.xlabel('Epoch')
    plt.ylabel('EPE (pixels)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    save_path = f"{save_dir}/{args.stage}_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved training curves to {save_path}")


def train(args):

    # print("Parameter Count: %d" % count_parameters(model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    model = model.to(device)
    # model = RAFT(args).to(device)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        #加载已有的模型权重
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    # model.cuda()
    model.train()
    args.stage = "train"

    # if args.datasets != 'chairs':
    #     model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model,train_loader)
    criterion = torch.nn.CrossEntropyLoss()

    # total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    # VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True

    mk_file('training_checkpoints')

    # train_losses, train_accuracies = train_with_plot(
    #     model, train_loader, optimizer, criterion, num_epochs=args.num_steps , device=device
    # )

    train_loss_history = []
    train_epe_history = []

    val_epe_history = []
    val_f1_history = []
    epochs = args.num_steps
    

    for epoch in range(epochs):

        epoch_loss_sum = 0.0
        epoch_epe_sum = 0.0
        epoch_batches = 0
        progress_bar = tqdm(train_loader,total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}",dynamic_ncols=True)

        for i_batch, data_blob in enumerate(progress_bar):

            
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.to(device, non_blocking=True)for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                noise1 = stdv * torch.randn_like(image1) 
                noise2 = stdv * torch.randn_like(image2)
                image1 = (image1 + noise1).clamp(0.0, 255.0)
                image2 = (image2 + noise2).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)            

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            epoch_loss_sum += loss.item()
            epoch_epe_sum += metrics['epe']
            epoch_batches += 1

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "epe": f"{metrics['epe']:.3f}"
            })

        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        avg_epe  = epoch_epe_sum  / max(epoch_batches, 1)
        train_loss_history.append(avg_loss)
        train_epe_history.append(avg_epe)
        print(f"[Epoch {epoch+1}/{epochs}] train loss = {avg_loss:.4f}, train EPE = {avg_epe:.3f}")

        model.eval()
        args.stage = "val"
        with torch.no_grad():
            val_results = {}
            for val_dataset in args.validation:
                if val_dataset == 'chairs':
                    val_results.update(evaluate.validate_chairs(model.module))
                elif val_dataset == 'sintel':
                    val_results.update(evaluate.validate_sintel(model.module))
                elif val_dataset == 'kitti':
                    val_results.update(evaluate.validate_kitti(model.module))
            logger.write_dict(val_results)

        val_epe_history.append(val_results.get('kitti', 0.0))
        val_f1_history.append(val_results.get('kitti-f1', 0.0))
        if 'kitti-epe' in val_results:
            print(f"          val kitti EPE = {val_results['kitti-epe']:.3f}")

        if epoch % 200 == 0 or epoch == epochs - 1:
            ckpt_path = f"training_checkpoints/epoch{epoch+1}_{args.name}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")
            
        model.train()
        args.stage = "train"
        if args.datasets != 'chairs':
            model.module.freeze_bn()



    logger.close()
    PATH = 'train_checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)
    print(f"[Final] Model saved to: {PATH}")
    plot_curves(train_loss_history, train_epe_history)
    plot_curves(val_epe_history, val_f1_history)

    return PATH


if __name__ == '__main__':
    args = SimpleNamespace(
        name='raft',
        datasets='kitti',              #datasets for training: 'chairs', 'things', 'sintel', 'kitti','chairs2'
        restore_ckpt=None,              # 或 'checkpoints/raft-things.pth'
        feat_type='dinov3',           #['small','basic','dinov3'] # 选择特征提取骨干网络
        dinov3_model='vitb16',      #['vitb16','vitl16'] # dinov3 模型类型
        validation='kitti', # 想在哪些验证集上评估
        stage = "train",

        lr=2e-5,
        num_steps=10,
        batch_size=64,
        crop_size=[320, 448],# FlyingChairs2 推荐使用这个尺寸
        image_size=[320, 1152],
        gpus=[0, 1, 2, 3, 4, 5],                       # 如果你只有一块卡就写 [0]
        mixed_precision=True,          # 显卡够的话也可以 True

        iters=12,
        wdecay=0.00005,
        epsilon=1e-8,
        clip=1.0,
        dropout=0.0,
        gamma=0.8,
        add_noise=False,                # 想用噪声增强就改成 True
        alternate_corr=False            # 没有 alt_cuda_corr 扩展就 False
    )


    torch.manual_seed(42)#the anwser of life, the universe and everything
    np.random.seed(42)

    if not os.path.isdir('train_checkpoints'):
        os.mkdir('train_checkpoints')

    args.datasets = 'chairs2'  # 选择训练数据集
    args.name ='raft_chairs2_stage1'
    args.restore_ckpt = None
    train(args)

    args.datasets = 'kitti'  # 选择训练数据集
    args.name ='raft_chairs2_kitti_ft'
    args.restore_ckpt = 'train_checkpoints/raft_chairs2_stage1.pth'
    args.num_steps = 200
    train(args)