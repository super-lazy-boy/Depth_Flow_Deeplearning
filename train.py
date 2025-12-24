# train.py
from __future__ import print_function, division
import sys
sys.path.append('core')

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from model.flowseek import FlowSeek
from model.datasets import fetch_dataloader, KITTI
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from model.utils.utils import InputPadder
from model.loss import sequence_loss

from tqdm import tqdm
from split_dataset import mk_file

try:
    from torch.amp import GradScaler
except:
    class GradScaler:
        def __init__(self, enabled=False): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass


MAX_FLOW = 400
SUM_FREQ = 100


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model, train_loader):
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.num_steps
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, total_steps=total_steps,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear'
    )
    return optimizer, scheduler


@torch.no_grad()
def validate_kitti_flowseek(model, kitti_root, iters=24):
    model.eval()
    val_dataset = KITTI(split='training', root=kitti_root)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        # IMPORTANT: KITTI dataset returns: img1,img2,flow,flow_valid,depth,depth_valid
        image1, image2, flow_gt, valid_gt, _, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        out = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(out['final'])[0].cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

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
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            self.running_loss[key] = self.running_loss.get(key, 0.0) + metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def plot_curves(values, title, xlabel="epoch", save_path="result", run_name="flowseek"):
    epochs = range(1, len(values) + 1)
    plt.figure(figsize=(10, 5))
    save_dir = f"{save_path}/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    plt.plot(epochs, values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()

    out_path = f"{save_dir}/{title}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved curve to {out_path}")


def depth_l1_loss(depth_pred, depth, depth_valid, eps=1e-6):
    """
    depth_pred: [B,1,H,W]
    depth:      [B,1,H,W]  (in this project, we use KITTI disparity as proxy target)
    depth_valid:[B,1,H,W]
    If depth_valid is all-zero (no supervision), return 0.
    """
    if depth is None or depth_valid is None:
        return depth_pred.new_tensor(0.0)

    mask = depth_valid > 0.5
    denom = mask.sum().clamp_min(1.0)

    # if all invalid: no supervised depth loss
    if denom.item() <= 1.0:
        return depth_pred.new_tensor(0.0)

    return (mask * (depth_pred - depth).abs()).sum() / denom


def train(args):
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(FlowSeek(args), device_ids=args.gpus).to(device)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.train()

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model, train_loader)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    mk_file('training_checkpoints')

    train_loss_history, train_epe_history = [], []
    val_epe_history, val_f1_history = [], []

    epochs = args.num_steps
    best_val_epe = float('inf')

    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        epoch_epe_sum = 0.0
        epoch_batches = 0

        progress_bar = tqdm(train_loader, total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)

        for _, data_blob in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)

            # IMPORTANT: must match datasets.py return order:
            # img1, img2, flow, flow_valid, depth, depth_valid
            image1, image2, flow, flow_valid, depth, depth_valid = data_blob
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            flow_valid = flow_valid.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            depth_valid = depth_valid.to(device, non_blocking=True)

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                noise1 = stdv * torch.randn_like(image1)
                noise2 = stdv * torch.randn_like(image2)
                image1 = (image1 + noise1).clamp(0.0, 255.0)
                image2 = (image2 + noise2).clamp(0.0, 255.0)

            output = model(image1, image2, iters=args.iters, flow_gt=flow, test_mode=False)
            flow_loss, metrics = sequence_loss(output, flow, flow_valid, gamma=args.gamma)

            dloss = depth_l1_loss(output['depth'], depth, depth_valid)
            loss = flow_loss + dloss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            epoch_loss_sum += float(loss.item())
            epoch_epe_sum += float(metrics['epe'])
            epoch_batches += 1

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "epe": f"{metrics['epe']:.3f}",
                "dloss": f"{dloss.item():.4f}",
            })

        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        avg_epe = epoch_epe_sum / max(epoch_batches, 1)
        train_loss_history.append(avg_loss)
        train_epe_history.append(avg_epe)
        print(f"[Epoch {epoch+1}/{epochs}] train loss = {avg_loss:.4f}, train EPE = {avg_epe:.3f}")

        model.eval()
        with torch.no_grad():
            val_results = validate_kitti_flowseek(model.module, kitti_root=args.paths['kitti'], iters=args.iters)
            logger.write_dict(val_results)

        val_epe_history.append(val_results.get('kitti-epe', 0.0))
        val_f1_history.append(val_results.get('kitti-f1', 0.0))
        print(f"    val kitti EPE = {val_results['kitti-epe']:.3f}, F1 = {val_results['kitti-f1']:.2f}")

        if epoch % 200 == 0 or epoch == epochs - 1:
            ckpt_path = f"training_checkpoints/epoch{epoch+1}_{args.name}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")

        current_val_epe = val_results.get('kitti-epe', float('inf'))
        if current_val_epe < best_val_epe:
            best_val_epe = current_val_epe
            torch.save(model.state_dict(), f"training_checkpoints/best_{args.name}.pth")

        model.train()

    logger.close()

    os.makedirs('train_checkpoints', exist_ok=True)
    final_path = f"train_checkpoints/{args.name}.pth"
    torch.save(model.state_dict(), final_path)
    print(f"[Final] Model saved to: {final_path}")

    plot_curves(train_loss_history, "train_loss", run_name=args.name)
    plot_curves(train_epe_history, "train_EPE", run_name=args.name)
    plot_curves(val_epe_history, "val_kitti_EPE", run_name=args.name)
    plot_curves(val_f1_history, "val_kitti_F1", run_name=args.name)

    return final_path


if __name__ == '__main__':
    args = SimpleNamespace(
        name="flowseek",
        dataset="kitti",
        stage="train",
        gpus=[0, 1],
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

        image_size=[384, 512],
        scale=0,
        batch_size=2,
        epsilon=1e-8,
        lr=4e-4,
        wdecay=1e-5,
        dropout=0,
        clip=1.0,
        gamma=0.85,
        num_steps=2,
        seed=42,
        mixed_precision=False,

        paths={
            'kitti': './data/KITTI/',
            'chairs': './data/FlyingChairs/data'
        },

        da_size="vitb"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
