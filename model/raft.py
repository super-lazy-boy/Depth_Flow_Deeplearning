import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.flow.update import BasicUpdateBlock, SmallUpdateBlock
from model.feat.extractor import BasicEncoder, SmallEncoder
from model.feat.corr import CorrBlock, AlternateCorrBlock
from model.feat.dinov3 import Dinov3Encoder
from model.flow.utils import bilinear_sampler, coords_grid, upflow8

# try:
#     autocast = torch.cuda.amp.autocast
# except:
#     # dummy autocast for PyTorch < 1.6
#     class autocast:
#         def __init__(self, enabled):
#             pass
#         def __enter__(self):
#             pass
#         def __exit__(self, *args):
#             pass

# 基于RSFT，针对作业要求修改
class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.feat_type == 'small':
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        elif args.feat_type == 'basic':
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
        
        elif args.feat_type == 'dinov3':
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if not hasattr(self.args, 'dropout'):
            self.args.dropout = 0.0

        if not hasattr(self.args, 'alternate_corr'):
            self.args.alternate_corr = False


        # feature network, context network, and update block
        if args.feat_type == 'small' :
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        elif args.feat_type in 'basic':
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        elif args.feat_type == 'dinov3':
            self.fnet = Dinov3Encoder(output_dim=256, model='vitb16', dropout=args.dropout)    
            self.cnet = Dinov3Encoder(output_dim=hdim+cdim, model='vitb16', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        #autocast = torch.amp.autocast('cuda', enabled=self.args.mixed_precision)

        # run the feature network
        with torch.amp.autocast('cuda', enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        B, C_img, H_img, W_img = image1.shape
        Hc = H_img // 8
        Wc = W_img // 8
        if fmap1.shape[2] != Hc or fmap1.shape[3] != Wc:
            # 这里使用双线性插值，把 [B, C, 46, 156] 变成 [B, C, 47, 156]
            fmap1 = F.interpolate(fmap1, size=(Hc, Wc), mode='bilinear', align_corners=False)
            fmap2 = F.interpolate(fmap2, size=(Hc, Wc), mode='bilinear', align_corners=False)

        assert fmap1.shape[2] == Hc and fmap1.shape[3] == Wc, \
            f"Feature map size {fmap1.shape[2:]} does not match expected {(Hc, Wc)}"    

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with torch.amp.autocast('cuda', enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
        cnet = cnet.float()

        # 如果 cnet 的空间分辨率和 (Hc, Wc) 不一致，一样插值过去
        if cnet.shape[2] != Hc or cnet.shape[3] != Wc:
            cnet = F.interpolate(cnet, size=(Hc, Wc), mode='bilinear', align_corners=False)

        # 防御式检查：确保最终 cnet 的 H、W 和 fmap / coords 一致
        assert cnet.shape[2] == Hc and cnet.shape[3] == Wc, \
            f"cnet spatial size {cnet.shape[2:]} != expected {(Hc, Wc)}"
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            # print("feat fmap shape:", fmap1.shape)  # 一般是 [B, C, Hc, Wc]
            # print("coords1 shape before flatten:", coords1.shape)

            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with torch.amp.autocast('cuda', enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
