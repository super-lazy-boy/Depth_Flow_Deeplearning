# flowseek.py
# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.depth_anything_v2.dpt import DepthAnythingV2
from model.update import BasicUpdateBlock
from model.corr import CorrBlock
from model.utils.utils import coords_grid, InputPadder
from model.extractor import ResNetFPN
from model.layer import conv3x3
from model.depth import DepthHead
from huggingface_hub import PyTorchModelHubMixin


class FlowSeek(nn.Module, PyTorchModelHubMixin):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.output_dim = args.dim * 2
        self.da_size = args.da_size

        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2

        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim,
                              norm_layer=nn.BatchNorm2d, init_weight=True)

        self.da_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        }

        self.dav2 = DepthAnythingV2(**self.da_model_configs[args.da_size])
        # NOTE: do not force .cuda() here; let outer .to(device) / DataParallel handle it
        ckpt = f'checkpoints/depth_anything_v2_{args.da_size}.pth'
        self.dav2.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        self.dav2.eval()
        for p in self.dav2.parameters():
            p.requires_grad = False

        self.merge_head = nn.Sequential(
            nn.Conv2d(self.da_model_configs[args.da_size]['features'],
                      self.da_model_configs[args.da_size]['features'] // 2 * 3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.da_model_configs[args.da_size]['features'] // 2 * 3,
                      self.da_model_configs[args.da_size]['features'] * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.da_model_configs[args.da_size]['features'] * 2,
                      self.da_model_configs[args.da_size]['features'] * 2, 3, stride=2, padding=1),
        )

        self.bnet = ResNetFPN(args, input_dim=16, output_dim=2 * self.args.dim,
                              norm_layer=nn.BatchNorm2d, init_weight=True)

        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)

        self.upsample_weight = nn.Sequential(
            nn.Conv2d(args.dim * 2, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(args.dim * 2, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1)
        )

        if args.iters > 0:
            self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim,
                                  norm_layer=nn.BatchNorm2d, init_weight=True)
            self.update_block = BasicUpdateBlock(args, hdim=args.dim * 2, cdim=args.dim * 2)

        self.depth_head = DepthHead(in_ch=1, hidden=32)

    def create_bases(self, disp):
        """
        disp: [B,1,H,W]  (here disp is DepthAnythingV2 output; treated as inverse-depth / disparity-like proxy)
        returns: M [B,16,H,W]
        """
        B, C, H, W = disp.shape
        assert C == 1
        device = disp.device

        cx = 0.5
        cy = 0.5

        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device)
        u, v = torch.meshgrid(xs, ys, indexing='xy')
        u = (u - cx).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        v = (v - cy).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

        aspect_ratio = W / H

        Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
        Tz = torch.cat([u, v], dim=1)

        Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1, 2, 3), keepdim=True)
        Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1, 2, 3), keepdim=True)
        Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1, 2, 3), keepdim=True)

        Tx = 2 * disp * Tx
        Ty = 2 * disp * Ty
        Tz = 2 * disp * Tz

        R1x = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
        R2x = torch.cat([u * v, v * v], dim=1)
        R1y = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        R2y = torch.cat([-u * u, -u * v], dim=1)
        Rz = torch.cat([-v / aspect_ratio, u * aspect_ratio], dim=1)

        R1x = R1x / torch.linalg.vector_norm(R1x, dim=(1, 2, 3), keepdim=True)
        R2x = R2x / torch.linalg.vector_norm(R2x, dim=(1, 2, 3), keepdim=True)
        R1y = R1y / torch.linalg.vector_norm(R1y, dim=(1, 2, 3), keepdim=True)
        R2y = R2y / torch.linalg.vector_norm(R2y, dim=(1, 2, 3), keepdim=True)
        Rz = Rz / torch.linalg.vector_norm(Rz, dim=(1, 2, 3), keepdim=True)

        M = torch.cat([Tx, Ty, Tz, R1x, R2x, R1y, R2y, Rz], dim=1)  # [B,16,H,W]
        return M

    def upsample_data(self, flow, info, mask):
        """Upsample [H/8,W/8] -> [H,W] using convex combination."""
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1).view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1).view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2).permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False, demo=False):
        N, _, H0, W0 = image1.shape
        device = image1.device

        if iters is None:
            iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H0, W0, device=device)

        # DepthAnything input normalization
        image1_res = F.interpolate(image1, (518, 518), mode="bilinear", align_corners=False) / 255.0
        image2_res = F.interpolate(image2, (518, 518), mode="bilinear", align_corners=False) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        # keep your original normalization behavior
        image1_res = image1_res / mean - std
        image2_res = image2_res / mean - std

        # dav2 is frozen
        im1_path1, depth1 = self.dav2.forward(image1_res.float())
        im2_path1, _ = self.dav2.forward(image2_res.float())

        depth1_ref = self.depth_head(depth1)
        depth_pred = F.interpolate(depth1_ref, (H0, W0), mode="bilinear", align_corners=False)

        im1_path1 = F.interpolate(im1_path1, (H0, W0), mode="bilinear", align_corners=False)
        im2_path1 = F.interpolate(im2_path1, (H0, W0), mode="bilinear", align_corners=False)

        bases1 = self.create_bases(F.interpolate(depth1, (H0, W0), mode="bilinear", align_corners=False))

        mono1 = self.merge_head(im1_path1)
        mono2 = self.merge_head(im2_path1)

        # RAFT-like scaling
        image1n = 2 * (image1 / 255.0) - 1.0
        image2n = 2 * (image2 / 255.0) - 1.0
        image1n = image1n.contiguous()
        image2n = image2n.contiguous()

        flow_predictions = []
        info_predictions = []

        # padding
        padder = InputPadder(image1n.shape)
        image1n, image2n = padder.pad(image1n, image2n)
        bases1_pad = padder.pad(bases1)

        if isinstance(bases1_pad, (list, tuple)):
            bases1 = bases1_pad[0]
        else:
            bases1 = bases1_pad

        N, _, H, W = image1n.shape
        dilation = torch.ones(N, 1, H // 8, W // 8, device=device)

        # context net
        cnet_inputs = torch.cat([image1n, image2n], dim=1)
        cnet = self.init_conv(self.cnet(cnet_inputs))
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        # bases net (IMPORTANT: keep batch dimension)
        bnet_inputs = bases1
        bnet = self.init_conv(self.bnet(bnet_inputs))
        netbases, ctxbases = torch.split(bnet, [self.args.dim, self.args.dim], dim=1)

        context = torch.cat((context, ctxbases), 1)
        net = torch.cat((net, netbases), 1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = 0.25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)

        if self.args.iters > 0:
            fmap1_8x = self.fnet(image1n)
            fmap2_8x = self.fnet(image2n)

            fmap1_8x = torch.cat((fmap1_8x, mono1), 1)
            fmap2_8x = torch.cat((fmap2_8x, mono2), 1)

            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

            for _ in range(iters):
                N, _, h8, w8 = flow_8x.shape
                flow_8x = flow_8x.detach()
                coords2 = (coords_grid(N, h8, w8, device=device) + flow_8x).detach()
                corr = corr_fn(coords2, dilation=dilation)

                net = self.update_block(net, context, corr, flow_8x)
                flow_update = self.flow_head(net)
                weight_update = 0.25 * self.upsample_weight(net)

                flow_8x = flow_8x + flow_update[:, :2]
                info_8x = flow_update[:, 2:]

                flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
                flow_predictions.append(flow_up)
                info_predictions.append(info_up)

        # unpad
        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if not test_mode:
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.args.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.args.var_max
                    var_min = self.args.var_min

                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]

                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)

                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(
                    term1.unsqueeze(1) - term2, dim=2
                )
                nf_predictions.append(nf_loss)

            return {
                'final': flow_predictions[-1],
                'flow': flow_predictions,
                'info': info_predictions,
                'nf': nf_predictions,
                'depth': depth_pred
            }
        else:
            return {
                'final': flow_predictions[-1],
                'flow': flow_predictions,
                'info': info_predictions,
                'nf': None,
                'depth': depth_pred
            }
