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
# from model.raft import ZAQ
from model.flowseek import FlowSeek
# import evaluate
from model.datasets import fetch_dataloader, KITTI

from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from model.utils.utils import InputPadder
from model.loss import sequence_loss

import matplotlib.pyplot as plt
from tqdm import tqdm

from split_dataset import mk_file
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


# def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
#     """ Loss function defined over sequence of flow predictions """

#     n_predictions = len(flow_preds)    
#     flow_loss = 0.0

#     # exlude invalid pixels and extremely large diplacements
#     mag = torch.sum(flow_gt**2, dim=1).sqrt()
#     valid = (valid >= 0.5) & (mag < max_flow)

#     for i in range(n_predictions):
#         i_weight = gamma**(n_predictions - i - 1)
#         i_loss = (flow_preds[i] - flow_gt).abs()
#         flow_loss += i_weight * (valid[:, None] * i_loss).mean()

#     epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
#     epe = epe.view(-1)[valid.view(-1)]

#     metrics = {
#         'epe': epe.mean().item(),
#         '1px': (epe < 1).float().mean().item(),
#         '3px': (epe < 3).float().mean().item(),
#         '5px': (epe < 5).float().mean().item(),
#     }

#     return flow_loss, metrics


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
    
@torch.no_grad()
def validate_kitti_flowseek(model, iters=24):
    model.eval()
    val_dataset = KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        out = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(out['final'])[0].cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        outlier = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(outlier[val].cpu().numpy())

    epe = float(np.mean(np.array(epe_list)))
    f1 = float(100 * np.mean(np.concatenate(out_list)))

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

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


def plot_curves(list, title, xlabel= "epoch", save_path="result"):
    """
    train_losses: 每个 epoch 的平均 loss
    train_epe_list: 每个 epoch 的平均 EPE（我们用它当作“精度指标”的反向度量，越低越好）
    """
    epochs = range(1, len(list)+1)
    ylabel = title

    plt.figure(figsize=(10,5))
    save_dir = f"{save_path}/{args.name}"
    if save_dir != "" and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loss 曲线
    plt.subplot(1,1,1)
    plt.plot(epochs, list, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    save_path = f"{save_dir}/{title}.png"
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved training curves to {save_path}")

def depth_l1_loss(depth_pred, depth, depth_valid, eps=1e-6):
    # depth_pred: [B,1,H,W]
    if depth.dim() == 3:  # [B,H,W]
        depth = depth.unsqueeze(1)  # [B,1,H,W]
    if depth_valid.dim() == 3:  # [B,H,W]
        depth_valid = depth_valid.unsqueeze(1)  # [B,1,H,W]
    mask = depth_valid > 0.5
    denom = mask.sum().clamp_min(1.0)
    return (mask * (depth_pred - depth).abs()).sum() / denom



def train(args):
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   

    # print("Parameter Count: %d" % count_parameters(model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = nn.DataParallel(ZAQ(args), device_ids=args.gpus)
    model = nn.DataParallel(FlowSeek(args), device_ids=args.gpus)
    model = model.to(device)
    # model = ZAQ(args).to(device)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        #加载已有的模型权重
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    # model.cuda()
    model.train()
   

    # if args.datasets != 'chairs':
    #     model.module.freeze_bn()

    train_loader = fetch_dataloader(args)
    # print("fetching one batch...", flush=True)
    # batch = next(iter(train_loader))
    # print("batch fetched OK", flush=True)
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

    best_val_epe = float('inf')
    

    for epoch in range(epochs):

        epoch_loss_sum = 0.0
        epoch_epe_sum = 0.0
        epoch_batches = 0
        progress_bar = tqdm(train_loader,total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}",dynamic_ncols=True)

        for i_batch, data_blob in enumerate(progress_bar):

            
            optimizer.zero_grad()
            image1, image2, flow, depth, flow_valid, depth_valid = [x.to(device, non_blocking=True)for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                noise1 = stdv * torch.randn_like(image1) 
                noise2 = stdv * torch.randn_like(image2)
                image1 = (image1 + noise1).clamp(0.0, 255.0)
                image2 = (image2 + noise2).clamp(0.0, 255.0)

            output = model(image1, image2, iters=args.iters, flow_gt=flow, test_mode=False)
            flow_loss, metrics = sequence_loss(output, flow, flow_valid, gamma=args.gamma)
            depth_loss = depth_l1_loss(output['depth'], depth, depth_valid)

            loss = flow_loss + depth_loss

            optimizer.zero_grad()
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
        
        with torch.no_grad():
            val_results = {}
            # for val_dataset in args.validation:
            #     if val_dataset == 'chairs':
            #         val_results.update(evaluate.validate_chairs(model.module))
            #     elif val_dataset == 'sintel':
            #         val_results.update(evaluate.validate_sintel(model.module))
            #     elif val_dataset == 'kitti':
            #         val_results.update(evaluate.validate_kitti(model.module))
            val_results.update(validate_kitti_flowseek(model.module, iters=args.iters))
            logger.write_dict(val_results)

        val_epe_history.append(val_results.get('kitti-epe', 0.0))
        val_f1_history.append(val_results.get('kitti-f1', 0.0))
        if 'kitti-epe' in val_results:
            print(f"    val kitti EPE = {val_results['kitti-epe']:.3f}")

        if epoch % 200 == 0 or epoch == epochs - 1:
            ckpt_path = f"training_checkpoints/epoch{epoch+1}_{args.name}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")

        current_val_epe = val_results.get('kitti-epe', float('inf'))
        if current_val_epe < best_val_epe:
            best_val_epe = current_val_epe
            torch.save(model.state_dict(), f"training_checkpoints/best_{args.name}.pth")

        model.train()
        
        # if args.dataset != 'chairs2':
        #     model.module.freeze_bn()


    logger.close()
    PATH = 'train_checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)
    print(f"[Final] Model saved to: {PATH}")
    args.stage ="train"
    plot_curves(train_loss_history,"trian_loss")
    plot_curves(train_epe_history,"train_EPE")
    plot_curves(val_epe_history,"val_kitti_EPE")
    plot_curves(val_f1_history,"val_kitti_F1")


    return PATH


if __name__ == '__main__':
    args = SimpleNamespace(
        name="flowseek",
        dataset="kitti",
        stage="train",
        gpus=[0],
        validation=['kitti'],
        use_var=True,
        var_min=0,
        var_max=10,
        pretrain="resnet34",
        initial_dim=64,
        block_dims=[64, 128, 256],
        radius=4,
        dim=128,
        num_blocks=2,
        iters=4,
        restore_ckpt=None,
        add_noise=False,

        image_size=[480, 640],
        scale=0,
        batch_size=1,
        epsilon=1e-8,
        lr=4e-4,
        wdecay=1e-5,
        dropout=0,
        clip=1.0,
        gamma=0.85,
        num_steps=2,
        seed=42,
        mixed_precision=False,
        paths={'kitti': './data/KITTI/'},

        da_size="vitb"
    )


    torch.manual_seed(42)#the anwser of life, the universe and everything
    np.random.seed(42)

    if not os.path.isdir('train_checkpoints'):
        os.mkdir('train_checkpoints')

    # # args.datasets = 'chairs2'  # 选择训练数据集
    # # args.name ='raft_chairs2_stage1'
    # args.restore_ckpt = None
    # train(args)

    # args.datasets = 'kitti'  # 选择训练数据集
    # # args.name ='raft_chairs2_kitti_ft'
    # args.name ='test'
    # # args.restore_ckpt = 'train_checkpoints/raft_chairs2_stage1.pth'
    # args.num_steps = 5
    train(args)