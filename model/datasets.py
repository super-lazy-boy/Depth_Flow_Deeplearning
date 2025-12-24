# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from model.utils import frame_utils
from model.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

def _read_kitti_disparity_png(path: str) -> np.ndarray:
    """
    KITTI disparity PNG (disp_noc_0) is typically 16-bit PNG where disp = value / 256.
    Returns float32 disparity map with shape [H, W].
    """
    disp = frame_utils.read_gen(path)          # usually PIL.Image
    disp = np.array(disp).astype(np.float32)
    # common KITTI encoding:
    disp = disp / 256.0
    return disp

def _pad_to(t: torch.Tensor, H: int, W: int):
    # t: [C,H,W]
    _, h, w = t.shape
    pad_h = H - h
    pad_w = W - w
    if pad_h == 0 and pad_w == 0:
        return t
    # pad format: (left, right, top, bottom)
    return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))

def pad_collate_fn(batch):
    """
    batch is list of tuples:
    (img1, img2, flow, flow_valid, depth, depth_valid)
    Each tensor is [C,H,W] except flow_valid might be [H,W] in some implementations.
    We pad all spatial tensors to max(H,W) in this batch.
    """
    imgs1, imgs2, flows, fvalids, depths, dvalids = zip(*batch)

    # find max H,W over all tensors that have 3 dims
    H = 0
    W = 0
    for t in list(imgs1) + list(imgs2) + list(flows) + list(depths) + list(dvalids):
        if t is None:
            continue
        if t.dim() == 3:
            H = max(H, t.shape[1])
            W = max(W, t.shape[2])

    imgs1 = torch.stack([_pad_to(t, H, W) for t in imgs1], dim=0)
    imgs2 = torch.stack([_pad_to(t, H, W) for t in imgs2], dim=0)
    flows = torch.stack([_pad_to(t, H, W) for t in flows], dim=0)

    # flow_valid may be [H,W] or [1,H,W] depending on your dataset code
    fvalid_list = []
    for v in fvalids:
        if v.dim() == 2:
            v = v.unsqueeze(0)
        fvalid_list.append(_pad_to(v, H, W))
    fvalids = torch.stack(fvalid_list, dim=0)

    depths = torch.stack([_pad_to(t, H, W) for t in depths], dim=0)
    dvalids = torch.stack([_pad_to(t, H, W) for t in dvalids], dim=0)

    return imgs1, imgs2, flows, fvalids, depths, dvalids


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.depth_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        # worker seed init
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        # ---------- flow ----------
        flow_valid = None
        if self.sparse:
            flow, flow_valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        # ---------- images ----------
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale -> 3ch
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # ---------- depth / disparity supervision ----------
        depth = None
        depth_valid = None
        if len(self.depth_list) > 0:
            dpath = self.depth_list[index]

            # KITTI disparity supervision (disp_noc_0) is 16-bit png: disp = val/256
            if isinstance(dpath, str) and (("disp_noc_0" in dpath) or ("disp_occ_0" in dpath)) and dpath.endswith(".png"):
                depth = _read_kitti_disparity_png(dpath)  # actually disparity
            else:
                depth_img = frame_utils.read_gen(dpath)
                depth = np.array(depth_img).astype(np.float32)

            depth_valid = (depth > 0).astype(np.float32)

        # augmentation (if enabled)
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, flow_valid = self.augmentor(img1, img2, flow, flow_valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        # to torch
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        # IMPORTANT: never return None for DataLoader default_collate
        # If depth is missing, return zeros + all-zero valid mask.
        if depth is None:
            H, W = img1.shape[1], img1.shape[2]
            depth = torch.zeros(1, H, W).float()
            depth_valid = torch.zeros(1, H, W).float()
        else:
            depth = torch.from_numpy(depth).unsqueeze(0).float()
            depth_valid = torch.from_numpy(depth_valid).unsqueeze(0).float()

        if flow_valid is not None:
            flow_valid = torch.from_numpy(flow_valid).float()
        else:
            flow_valid = ((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float()

        # Return order MUST match train.py unpacking
        # img1, img2, flow, flow_valid, depth, depth_valid
        return img1, img2, flow, flow_valid, depth, depth_valid


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        if len(self.depth_list) > 0:
            self.depth_list = v * self.depth_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            self.depth_list = sorted(glob(osp.join(root, 'disp_noc_0/*_10.png')))  


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.dataset == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.dataset == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.dataset == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training', root=args.paths['kitti'])

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True, collate_fn=pad_collate_fn)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

