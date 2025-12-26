# KITTI没有公开的验证集标签，这里使用训练集进行验证

import os
from shutil import copy,rmtree
import random
import argparse
import os
import random
import shutil
from glob import glob
from pathlib import Path

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # if the folder exists, delete it and create a new one
        rmtree(file_path)
    os.makedirs(file_path)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data/KITTI",
                    help="KITTI root dir that contains 'training/'")
    ap.add_argument("--ratio", type=float, default=0.9,
                    help="train split ratio (e.g., 0.9 -> 90% train, 10% val)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlink (default: symlink)")
    return ap.parse_args()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    args = parse_args()
    random.seed(args.seed)

    training_dir = Path(args.data_root) / "training"
    assert training_dir.exists(), f"{training_dir} not found"

    # Required subfolders
    subdirs = ["image_2", "flow_occ", "disp_noc_0", "calib_cam_to_cam"]

    for s in subdirs:
        assert (training_dir / s).exists(), f"Missing {s} in training"

    # Collect frame ids from image_2/*_10.png
    img10 = sorted(glob(str(training_dir / "image_2" / "*_10.png")))
    frame_ids = [Path(p).stem.split("_")[0] for p in img10]

    assert len(frame_ids) > 0, "No frames found"

    # Shuffle and split
    idx = list(range(len(frame_ids)))
    random.shuffle(idx)
    n_train = int(len(idx) * args.ratio)
    train_idx = set(idx[:n_train])
    val_idx = set(idx[n_train:])

    out_root = Path(args.data_root + "_split")
    train_out = out_root / "train"
    val_out = out_root / "val"

    for base in [train_out, val_out]:
        for s in subdirs:
            ensure_dir(base / s)

    def link_or_copy(src, dst):
        if args.copy:
            shutil.copy2(src, dst)
        else:
            # create symlink
            if os.path.exists(dst):
                return
            os.symlink(os.path.abspath(src), dst)

    for i, fid in enumerate(frame_ids):
        target = train_out if i in train_idx else val_out

        # image_2
        for suffix in ["_10.png", "_11.png"]:
            src = training_dir / "image_2" / f"{fid}{suffix}"
            dst = target / "image_2" / f"{fid}{suffix}"
            link_or_copy(src, dst)

        # flow
        src = training_dir / "flow_occ" / f"{fid}_10.png"
        dst = target / "flow_occ" / f"{fid}_10.png"
        link_or_copy(src, dst)

        # disparity
        src = training_dir / "disp_noc_0" / f"{fid}_10.png"
        dst = target / "disp_noc_0" / f"{fid}_10.png"
        link_or_copy(src, dst)

        # calibration
        src = training_dir / "calib_cam_to_cam" / f"{fid}.txt"
        dst = target / "calib_cam_to_cam" / f"{fid}.txt"
        link_or_copy(src, dst)

    print(f"Split done.")
    print(f"Train: {len(train_idx)} samples")
    print(f"Val  : {len(val_idx)} samples")
    print(f"Output dir: {out_root}")

if __name__ == "__main__":
    main()
