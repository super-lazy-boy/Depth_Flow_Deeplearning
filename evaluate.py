# evaluate.py (REPLACEMENT VERSION)
import os
import math
import numpy as np
from types import SimpleNamespace
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# IMPORTANT:
# This assumes you run evaluate.py at project root where:
#   - datasets.py exists (module name: datasets)
#   - model/flowseek.py exists and can be imported as model.flowseek
import model.datasets as datasets_module
from model.flowseek import FlowSeek


# -----------------------------
# Utils: robust state_dict load
# -----------------------------
def _strip_module_prefix(state_dict):
    """If checkpoint was saved from DataParallel, keys start with 'module.'; strip it."""
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    # Some projects save {"state_dict": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    ckpt = _strip_module_prefix(ckpt)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    if len(missing) > 0:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if len(unexpected) > 0:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")


# -----------------------------
# Utils: flow visualization (RAFT-style color wheel)
# -----------------------------
def make_colorwheel():
    """
    Color wheel as used in many optical flow papers / RAFT codebase style.
    Returns: [ncols, 3] in uint8.
    """
    # RY, YG, GC, CB, BM, MR
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(RY) / RY).astype(np.uint8)
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG).astype(np.uint8)
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(GC) / GC).astype(np.uint8)
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB).astype(np.uint8)
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(BM) / BM).astype(np.uint8)
    col += BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR).astype(np.uint8)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Convert flow to RGB image using color wheel.
    flow_uv: [H, W, 2] float32
    """
    flow = flow_uv.copy()
    if clip_flow is not None:
        flow = np.clip(flow, -clip_flow, clip_flow)

    u = flow[..., 0]
    v = flow[..., 1]

    rad = np.sqrt(u * u + v * v)
    rad_max = np.max(rad) + 1e-5

    u = u / rad_max
    v = v / rad_max

    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        col0 = colorwheel[k0, i] / 255.0
        col1 = colorwheel[k1, i] / 255.0
        col = (1 - f) * col0 + f * col1

        # decrease saturation with radius
        col = 1 - rad / (rad_max) * (1 - col)
        img[..., i] = np.floor(255 * col).astype(np.uint8)

    if convert_to_bgr:
        img = img[..., ::-1]
    return img


# -----------------------------
# Utils: depth visualization (paper-style colormap)
# -----------------------------
def depth_to_colormap(depth, valid_mask=None, dmin=None, dmax=None):
    """
    depth: [H,W] float32
    valid_mask: [H,W] bool
    Returns uint8 RGB.
    """
    import matplotlib.cm as cm

    dep = depth.copy()
    if valid_mask is None:
        valid_mask = np.isfinite(dep) & (dep > 0)

    if dmin is None:
        dmin = np.percentile(dep[valid_mask], 1) if np.any(valid_mask) else 0.0
    if dmax is None:
        dmax = np.percentile(dep[valid_mask], 99) if np.any(valid_mask) else 1.0
    dmax = max(dmax, dmin + 1e-6)

    dep = np.clip(dep, dmin, dmax)
    dep_norm = (dep - dmin) / (dmax - dmin + 1e-6)

    # common paper-like colormap: magma / plasma / inferno
    cmap = cm.get_cmap("magma")
    colored = cmap(dep_norm)[:, :, :3]  # [H,W,3] float
    colored[~valid_mask] = 0.0
    colored = (colored * 255.0).astype(np.uint8)
    return colored


def tensor_image_to_uint8(img_t):
    """
    img_t: torch tensor [3,H,W], values in [0,255] float
    """
    img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_png(path, arr_uint8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr_uint8).save(path)


def concat_horiz(img_list):
    """Concatenate list of HxWx3 uint8 images horizontally with same height."""
    H = img_list[0].shape[0]
    outs = []
    for im in img_list:
        if im.shape[0] != H:
            # resize to match height
            w = int(im.shape[1] * (H / im.shape[0]))
            im = np.array(Image.fromarray(im).resize((w, H), resample=Image.BILINEAR))
        outs.append(im)
    return np.concatenate(outs, axis=1)


# -----------------------------
# Build args (MUST include FlowSeek dependencies)
# -----------------------------
def build_args():
    base = os.path.dirname(__file__)

    args = SimpleNamespace(
        # paths
        kitti_root=os.path.join(base, "data", "KITTI_split"),     # contains training/ testing/
        split="testing",                                          # use testing
        ckpt_path=os.path.join(base, "train_checkpoints", "deeplearning_depth.pth"),

        # inference
        batch_size=1,
        num_workers=2,
        gpus=[0],
        mixed_precision=True,
        iters=4,              # must match training (or you can increase for better quality)
        save_dir=os.path.join(base, "result_test", "deeplearning_depth"),

        # FlowSeek / ResNetFPN required hyperparams (copy from train.py defaults)
        pretrain="resnet34",
        initial_dim=64,
        block_dims=[64, 128, 256],

        radius=4,
        dim=128,
        num_blocks=2,

        # flow uncertainty branch configs used in forward (safe defaults)
        use_var=True,
        var_min=0,
        var_max=10,

        # DepthAnythingV2 backbone size used in FlowSeek
        da_size="vitb",        # must match available checkpoints: checkpoints/depth_anything_v2_vitb.pth

        # (kept for compatibility)
        dataset="kitti",
        stage="test",
    )
    return args


# -----------------------------
# Data loader
# -----------------------------
def build_kitti_test_loader(args):
    # datasets.py defines KITTI(root=... , split=...)
    ds = datasets_module.KITTI(split=args.split, root=args.kitti_root)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"[INFO] KITTI {args.split} samples: {len(ds)}")
    return loader


# -----------------------------
# Main inference
# -----------------------------
@torch.no_grad()
def run_kitti_testing_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    loader = build_kitti_test_loader(args)

    # Build model
    model = FlowSeek(args).to(device)
    model.eval()

    # Load weights
    load_checkpoint(model, args.ckpt_path, device)

    # output dirs
    out_rgb = os.path.join(args.save_dir, "rgb")
    out_flow = os.path.join(args.save_dir, "flow")
    out_depth = os.path.join(args.save_dir, "depth")
    out_triplet = os.path.join(args.save_dir, "triplet")

    os.makedirs(out_rgb, exist_ok=True)
    os.makedirs(out_flow, exist_ok=True)
    os.makedirs(out_depth, exist_ok=True)
    os.makedirs(out_triplet, exist_ok=True)

    for i, batch in enumerate(loader):
        # KITTI in test mode returns (img1, img2, extra_info)
        # but if split contains gt it may return 6-tuple. Handle both robustly.
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            image1, image2, extra = batch
            frame_name = extra[0][0] if isinstance(extra, (list, tuple)) else f"{i:06d}_10.png"
        else:
            # supervised variant: img1,img2,flow,flow_valid,depth,depth_valid
            image1, image2 = batch[0], batch[1]
            frame_name = f"{i:06d}_10.png"

        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)

        # forward
        out = model(image1, image2, iters=args.iters, test_mode=True)

        # Flow: out['final'] is [B,2,H,W]
        flow = out["final"][0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        flow_img = flow_to_image(flow)

        # Depth: out['depth'] is [B,1,H,W]
        depth = out.get("depth", None)
        if depth is not None:
            dep = depth[0, 0].detach().cpu().numpy().astype(np.float32)
            dep_img = depth_to_colormap(dep, valid_mask=np.isfinite(dep) & (dep > 0))
        else:
            dep_img = np.zeros((flow_img.shape[0], flow_img.shape[1], 3), dtype=np.uint8)

        # RGB for reference
        rgb_img = tensor_image_to_uint8(image1[0])

        # save
        stem = os.path.splitext(frame_name)[0]
        save_png(os.path.join(out_rgb, f"{stem}_rgb.png"), rgb_img)
        save_png(os.path.join(out_flow, f"{stem}_flow.png"), flow_img)
        save_png(os.path.join(out_depth, f"{stem}_depth.png"), dep_img)

        trip = concat_horiz([rgb_img, flow_img, dep_img])
        save_png(os.path.join(out_triplet, f"{stem}_triplet.png"), trip)

        if (i + 1) % 5 == 0 or (i + 1) == len(loader):
            print(f"[INFO] Processed {i+1}/{len(loader)}")


if __name__ == "__main__":
    args = build_args()
    run_kitti_testing_inference(args)
