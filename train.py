from __future__ import print_function, division
import sys
sys.path.append('core')

import os
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model.flowseek import FlowSeek
from model.datasets import fetch_dataloader, KITTI
from model.utils.utils import InputPadder
from model.loss import sequence_loss

from tqdm import tqdm
from split_dataset import mk_file

try:
    from torch.amp import GradScaler
except Exception:
    class GradScaler:
        def __init__(self, enabled=False): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass


MAX_FLOW = 400


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


def plot_curve(x, y, title, save_dir, xlabel="epoch", ylabel=None):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel is not None else title)
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{title}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_all_curves(history_dict, save_dir, run_name):
    """
    history_dict:
      {
        "train/flow_loss": [...],
        "val/flow_loss": [...],
        ...
      }
    """
    out_dir = os.path.join(save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    epochs = list(range(1, len(next(iter(history_dict.values()))) + 1))

    saved = []
    for k, v in history_dict.items():
        title = k.replace("/", "_")
        saved.append(plot_curve(epochs, v, title=title, save_dir=out_dir, xlabel="epoch"))
    print(f"[INFO] Curves saved to: {out_dir}")
    for p in saved:
        print(f"  - {p}")


def depth_l1_loss(depth_pred, depth, depth_valid, eps=1e-6):
    """
    depth_pred: [B,1,H,W]
    depth:      [B,1,H,W]
    depth_valid:[B,1,H,W] or [B,H,W]
    """
    if depth is None or depth_valid is None:
        return depth_pred.new_tensor(0.0)

    if depth_valid.dim() == 3:
        depth_valid = depth_valid.unsqueeze(1)

    mask = depth_valid > 0.5
    denom = mask.sum().clamp_min(1.0)
    if denom.item() <= 1.0:
        return depth_pred.new_tensor(0.0)

    # 尺寸不一致则对齐
    if depth_pred.shape[-2:] != depth.shape[-2:]:
        depth_pred = F.interpolate(depth_pred, size=depth.shape[-2:], mode="bilinear", align_corners=False)

    return (mask * (depth_pred - depth).abs()).sum() / denom


@torch.no_grad()
def depth_metrics(depth_pred, depth_gt, depth_valid, eps=1e-6):
    """
    返回：mae, rmse, abs_rel
    depth_pred: [B,1,H,W]
    depth_gt:   [B,1,H,W]
    depth_valid:[B,1,H,W] or [B,H,W]
    """
    if depth_gt is None or depth_valid is None:
        return {"depth_mae": 0.0, "depth_rmse": 0.0, "depth_abs_rel": 0.0}

    if depth_valid.dim() == 3:
        depth_valid = depth_valid.unsqueeze(1)

    if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
        depth_pred = F.interpolate(depth_pred, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)

    mask = depth_valid > 0.5
    denom = mask.sum().clamp_min(1.0)

    if denom.item() <= 1.0:
        return {"depth_mae": 0.0, "depth_rmse": 0.0, "depth_abs_rel": 0.0}

    diff = (depth_pred - depth_gt)
    mae = (mask * diff.abs()).sum() / denom
    rmse = torch.sqrt((mask * diff.pow(2)).sum() / denom)

    abs_rel = (mask * (diff.abs() / (depth_gt.abs() + eps))).sum() / denom

    return {
        "depth_mae": float(mae.item()),
        "depth_rmse": float(rmse.item()),
        "depth_abs_rel": float(abs_rel.item())
    }


def build_kitti_loader(kitti_root, split, batch_size, image_size, num_workers=4):
    """
    使用你拆分后的目录：
      kitti_root/
        training/
        testing /
    split 取 "training" 或 "testing"
    """
    ds = KITTI(split=split, root=kitti_root)  # 你已改造 KITTI 类返回 depth, depth_valid
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "training"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "training")
    )
    return loader


@torch.no_grad()
def validate_one_epoch(model, val_loader, device, iters, epoch_idx, epochs_total):
    """
    逐 batch 评估：flow_loss, epe, f1, depth_loss, depth_mae, depth_rmse, depth_abs_rel
    注意：validation 不进行反传，仅 forward + metrics
    """
    model.eval()

    flow_loss_sum = 0.0
    epe_sum = 0.0
    f1_sum = 0.0

    depth_loss_sum = 0.0
    depth_mae_sum = 0.0
    depth_rmse_sum = 0.0
    depth_abs_rel_sum = 0.0

    n_batches = 0

    pbar = tqdm(val_loader, total=len(val_loader),
                desc=f"Val   {epoch_idx+1}/{epochs_total}", dynamic_ncols=True)

    for data_blob in pbar:
        if len(data_blob) == 3:
            # test 模式：只有 image1, image2, extra_info
            image1, image2, extra_info = data_blob
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)

            # 仅 forward（如果你需要保存预测可在这里做）
            _ = model(image1, image2, iters=iters, flow_gt=None, test_mode=True)
            continue
        # IMPORTANT: dataset returns: img1,img2,flow,flow_valid,depth,depth_valid
        image1, image2, flow, flow_valid, depth, depth_valid = data_blob

        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        flow_valid = flow_valid.to(device, non_blocking=True)

        depth = depth.to(device, non_blocking=True) if depth is not None else None
        depth_valid = depth_valid.to(device, non_blocking=True) if depth_valid is not None else None

        out = model(image1, image2, iters=iters, flow_gt=flow, test_mode=False)

        # flow metrics (sequence_loss already returns epe/f1)
        flow_loss, metrics = sequence_loss(out, flow, flow_valid, gamma=0.85)

        dloss = depth_l1_loss(out.get("depth", None), depth, depth_valid)
        dmet = depth_metrics(out.get("depth", None), depth, depth_valid)

        flow_loss_sum += float(flow_loss.item())
        epe_sum += float(metrics.get("epe", 0.0))
        f1_sum += float(metrics.get("f1", 0.0))

        depth_loss_sum += float(dloss.item())
        depth_mae_sum += float(dmet["depth_mae"])
        depth_rmse_sum += float(dmet["depth_rmse"])
        depth_abs_rel_sum += float(dmet["depth_abs_rel"])

        n_batches += 1

        pbar.set_postfix({
            "flow_loss": f"{flow_loss.item():.4f}",
            "epe": f"{metrics.get('epe', 0.0):.3f}",
            "f1": f"{metrics.get('f1', 0.0):.2f}",
            "d_loss": f"{dloss.item():.4f}",
            "d_mae": f"{dmet['depth_mae']:.3f}"
        })

    denom = max(n_batches, 1)
    results = {
        "val/flow_loss": flow_loss_sum / denom,
        "val/epe": epe_sum / denom,
        "val/f1": f1_sum / denom,
        "val/depth_loss": depth_loss_sum / denom,
        "val/depth_mae": depth_mae_sum / denom,
        "val/depth_rmse": depth_rmse_sum / denom,
        "val/depth_abs_rel": depth_abs_rel_sum / denom,
    }

    return results


def train(args):
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(FlowSeek(args), device_ids=args.gpus).to(device)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    # ===== Data =====
    # train_loader：沿用你原有 fetch_dataloader（包含 augment）
    train_loader = fetch_dataloader(args)

    # val_loader：使用你拆分的 KITTI_split/val（不做增强）
    # args.paths['kitti'] 指向 ./data/KITTI_split
    val_loader = build_kitti_loader(
        kitti_root=args.paths['kitti'],
        split="testing",
        batch_size=1,                 # 验证用 1 最稳，避免 batch 内尺寸差异
        image_size=args.image_size,
        num_workers=2
    )

    optimizer, scheduler = fetch_optimizer(args, model, train_loader)
    scaler = GradScaler(enabled=args.mixed_precision)

    mk_file('training_checkpoints')

    epochs = args.num_steps
    best_val_epe = float('inf')

    # 历史曲线记录（每 epoch 一个点）
    history = {
        "train/flow_loss": [],
        "train/epe": [],
        "train/f1": [],
        "train/depth_loss": [],
        "train/depth_mae": [],
        "train/depth_rmse": [],
        "train/depth_abs_rel": [],

        "val/flow_loss": [],
        "val/epe": [],
        "val/f1": [],
        "val/depth_loss": [],
        "val/depth_mae": [],
        "val/depth_rmse": [],
        "val/depth_abs_rel": [],
    }

    for epoch in range(epochs):
        model.train()

        # epoch 累积
        flow_loss_sum = 0.0
        epe_sum = 0.0
        f1_sum = 0.0

        depth_loss_sum = 0.0
        depth_mae_sum = 0.0
        depth_rmse_sum = 0.0
        depth_abs_rel_sum = 0.0

        n_batches = 0

        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Train {epoch+1}/{epochs}", dynamic_ncols=True)

        for data_blob in pbar:
            optimizer.zero_grad(set_to_none=True)

            image1, image2, flow, flow_valid, depth, depth_valid = data_blob
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            flow_valid = flow_valid.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True) if depth is not None else None
            depth_valid = depth_valid.to(device, non_blocking=True) if depth_valid is not None else None

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                noise1 = stdv * torch.randn_like(image1)
                noise2 = stdv * torch.randn_like(image2)
                image1 = (image1 + noise1).clamp(0.0, 255.0)
                image2 = (image2 + noise2).clamp(0.0, 255.0)

            out = model(image1, image2, iters=args.iters, flow_gt=flow, test_mode=False)

            flow_loss, metrics = sequence_loss(out, flow, flow_valid, gamma=args.gamma)
            dloss = depth_l1_loss(out.get("depth", None), depth, depth_valid)

            # 深度指标（训练也统计，便于曲线）
            dmet = depth_metrics(out.get("depth", None), depth, depth_valid)

            loss = args.flow_weight*flow_loss + args.depth_weight * dloss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            flow_loss_sum += float(flow_loss.item())
            epe_sum += float(metrics.get("epe", 0.0))
            f1_sum += float(metrics.get("f1", 0.0))

            depth_loss_sum += float(dloss.item())
            depth_mae_sum += float(dmet["depth_mae"])
            depth_rmse_sum += float(dmet["depth_rmse"])
            depth_abs_rel_sum += float(dmet["depth_abs_rel"])

            n_batches += 1

            pbar.set_postfix({
                "flow_loss": f"{flow_loss.item():.4f}",
                "epe": f"{metrics.get('epe', 0.0):.3f}",
                "f1": f"{metrics.get('f1', 0.0):.2f}",
                "d_loss": f"{dloss.item():.4f}",
                "d_mae": f"{dmet['depth_mae']:.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        denom = max(n_batches, 1)
        train_results = {
            "train/flow_loss": flow_loss_sum / denom,
            "train/epe": epe_sum / denom,
            "train/f1": f1_sum / denom,
            "train/depth_loss": depth_loss_sum / denom,
            "train/depth_mae": depth_mae_sum / denom,
            "train/depth_rmse": depth_rmse_sum / denom,
            "train/depth_abs_rel": depth_abs_rel_sum / denom,
        }

        # 训练 epoch 结束：只打印一次
        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train | flow_loss={train_results['train/flow_loss']:.4f}, "
            f"epe={train_results['train/epe']:.3f}, f1={train_results['train/f1']:.2f} | "
            f"depth_loss={train_results['train/depth_loss']:.4f}, "
            f"mae={train_results['train/depth_mae']:.3f}, "
            f"rmse={train_results['train/depth_rmse']:.3f}, "
            f"abs_rel={train_results['train/depth_abs_rel']:.3f}"
        )

        # ===== Validation =====
        val_results = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            iters=args.iters,
            epoch_idx=epoch,
            epochs_total=epochs
        )

        # 验证 epoch 结束：只打印一次
        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Val   | flow_loss={val_results['val/flow_loss']:.4f}, "
            f"epe={val_results['val/epe']:.3f}, f1={val_results['val/f1']:.2f} | "
            f"depth_loss={val_results['val/depth_loss']:.4f}, "
            f"mae={val_results['val/depth_mae']:.3f}, "
            f"rmse={val_results['val/depth_rmse']:.3f}, "
            f"abs_rel={val_results['val/depth_abs_rel']:.3f}"
        )

        # 记录历史
        for k, v in train_results.items():
            history[k].append(v)
        for k, v in val_results.items():
            history[k].append(v)

        # checkpoint（更合理：每个 epoch 保存 last；best 按 val/flow_loss）
        ckpt_last = f"training_checkpoints/{args.name}/last_{args.name}.pth"
        torch.save(model.state_dict(), ckpt_last)

        best_val_loss = float("inf") 
        current_val_loss = val_results.get("val/flow_loss", float("inf"))*args.flow_weight+val_results.get("val/depth_loss", float("inf"))*args.depth_weight
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            ckpt_best = f"training_checkpoints/{args.name}/best_{args.name}.pth"
            torch.save(model.state_dict(), ckpt_best)
            print(f"[Checkpoint] Best updated: val_loss={best_val_loss:.3f} -> {ckpt_best}")

    # 保存最终模型
    os.makedirs('train_checkpoints', exist_ok=True)
    final_path = f"train_checkpoints/{args.name}.pth"
    torch.save(model.state_dict(), final_path)
    print(f"[Final] Model saved to: {final_path}")

    # 保存曲线
    save_all_curves(history, save_dir="result", run_name=args.name)

    return final_path


if __name__ == '__main__':
    args = SimpleNamespace(
        name="deeplearning_flow",
        dataset="kitti",
        stage="train",
        gpus=[0,1],                 # 建议先单卡跑通；多卡再开
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
        batch_size=4,
        epsilon=1e-8,
        lr=4e-4,
        wdecay=1e-5,
        dropout=0,
        clip=1.0,
        gamma=0.85,

        num_steps=200,  # 训练 epoch 数
        seed=42,
        mixed_precision=False,

        flow_weight=1.0,   # 光流 loss 权重；可从 0.1~1.0 调参
        depth_weight=1.0,  # 深度 loss 权重；可从 0.1~1.0 调参

        paths={
            'kitti': './data/KITTI_split',   # 这里必须是 split 后的根目录
            'chairs': './data/FlyingChairs/data'
        },

        da_size="vitb"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.mixed_precision = True

    args.name = "deeplearning_flow"
    args.flow_weight=1.0
    args.depth_weight=0.0
    train(args)
    args.name = "deeplearning_depth"
    args.flow_weight=0.0
    args.depth_weight=1.0
    train(args)
