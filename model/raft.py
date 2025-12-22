# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from model.update import BasicUpdateBlock, SmallUpdateBlock
# from model.extractor import BasicEncoder, SmallEncoder
# from model.corr import CorrBlock, AlternateCorrBlock
# from model.dinov3 import Dinov3Encoder
# from model.utils.utils import bilinear_sampler, coords_grid, upflow8, InputPadder
# from model.adapter import Adapter
# from model.extractor import ResNetFPN

# from model.depth_anything_v2.dpt import DepthAnythingV2
# ckpt_path = "checkpoints/depth_anything_v2_vitb.pth"

# # try:
# #     autocast = torch.cuda.amp.autocast
# # except:
# #     # dummy autocast for PyTorch < 1.6
# #     class autocast:
# #         def __init__(self, enabled):
# #             pass
# #         def __enter__(self):
# #             pass
# #         def __exit__(self, *args):
# #             pass

# # 基于RSFT，针对作业要求修改
# class ZAQ(nn.Module):
#     def __init__(self, args):
#         super(ZAQ, self).__init__()
#         self.args = args

#         if args.feat_type == 'small':
#             self.hidden_dim = hdim = 96
#             self.context_dim = cdim = 64
#             args.corr_levels = 4
#             args.corr_radius = 3
        
#         elif args.feat_type == 'basic':
#             self.hidden_dim = hdim = 128
#             self.context_dim = cdim = 128
#             args.corr_levels = 4
#             args.corr_radius = 4
        
#         elif args.feat_type == 'dinov3':
#             self.hidden_dim = hdim = 128
#             self.context_dim = cdim = 128
#             args.corr_levels = 4
#             args.corr_radius = 4

#         if not hasattr(self.args, 'dropout'):
#             self.args.dropout = 0.0

#         if not hasattr(self.args, 'alternate_corr'):
#             self.args.alternate_corr = False


#         # feature network, context network, and update block
#         if args.feat_type == 'small' :
#             self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
#             self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
#             self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

#         elif args.feat_type in 'basic':
#             self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
#             self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
#             self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

#         elif args.feat_type == 'dinov3':
#             self.fnet = Dinov3Encoder(output_dim=256, model='vitb16', dropout=args.dropout)    
#             self.cnet = Dinov3Encoder(output_dim=hdim+cdim, model='vitb16', dropout=args.dropout)
#             self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

#         # Depth Anything V2 模型部分
#         self.da_model_configs = {
#             'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#             'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#         }
#         self.merge_head = nn.Sequential(
#             nn.Conv2d(self.da_model_configs[args.da_size]['features'], self.da_model_configs[args.da_size]['features']//2*3, 3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.da_model_configs[args.da_size]['features']//2*3, self.da_model_configs[args.da_size]['features']*2, 3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.da_model_configs[args.da_size]['features']*2, self.da_model_configs[args.da_size]['features']*2, 3, stride=2, padding=1),
#         )

#         self.flow_adapter = Adapter(channels=256)

#         self.depth_model = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
#         # self.depth_model.load_state_dict(torch.load(ckpt_path), strict=False)
#         # self.depth_model.to(self.args.device)
#         # model = nn.DataParallel(model, device_ids=args.gpus)
#         self.depth_model.eval()  # Set to evaluation mode
#         for param in self.depth_model.parameters():
#             param.requires_grad = False  # Freeze parameters

#         in_dim = 1
#         hidden = 32 
#         self.depthhead = nn.Sequential(
#             nn.Conv2d(in_dim, hidden, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, 1, 3, padding=1),
#         )
#         self.bnet = ResNetFPN(args, input_dim=16, output_dim=hdim+cdim, norm_layer=nn.BatchNorm2d, init_weight=False)

#         # self._hook = self.depth_model.depth_head.scratch.output_conv1.register_forward_hook(self._save_feat)

#     def create_bases(self, disp):
#         B, C, H, W = disp.shape
#         assert C == 1
#         cx = 0.5
#         cy = 0.5

#         ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
#         xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
#         u, v = torch.meshgrid(xs, ys, indexing='xy')
#         u = u - cx
#         v = v - cy
#         u = u.unsqueeze(0).unsqueeze(0)
#         v = v.unsqueeze(0).unsqueeze(0)
#         u = u.repeat(B, 1, 1, 1).cuda()
#         v = v.repeat(B, 1, 1, 1).cuda()

#         aspect_ratio = W / H

#         Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
#         Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
#         Tz = torch.cat([u, v], dim=1)

#         Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1,2,3), keepdim=True)
#         Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1,2,3), keepdim=True)
#         Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1,2,3), keepdim=True)
        
#         Tx = 2 * disp * Tx
#         Ty = 2 * disp * Ty
#         Tz = 2 * disp * Tz

#         R1x = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
#         R2x = torch.cat([u * v, v * v], dim=1)
#         R1y = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
#         R2y = torch.cat([-u * u, -u * v], dim=1)
#         Rz =  torch.cat([-v / aspect_ratio, u * aspect_ratio], dim=1)

#         R1x = R1x / torch.linalg.vector_norm(R1x, dim=(1,2,3), keepdim=True)
#         R2x = R2x / torch.linalg.vector_norm(R2x, dim=(1,2,3), keepdim=True)
#         R1y = R1y / torch.linalg.vector_norm(R1y, dim=(1,2,3), keepdim=True)
#         R2y = R2y / torch.linalg.vector_norm(R2y, dim=(1,2,3), keepdim=True)
#         Rz =  Rz  / torch.linalg.vector_norm(Rz,  dim=(1,2,3), keepdim=True)
        
#         M = torch.cat([Tx, Ty, Tz, R1x, R2x, R1y, R2y, Rz], dim=1) # Bx(8x2)xHxW
#         return M

#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()

#     def initialize_flow(self, img):
#         """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
#         N, C, H, W = img.shape
#         coords0 = coords_grid(N, H//8, W//8, device=img.device)
#         coords1 = coords_grid(N, H//8, W//8, device=img.device)

#         # optical flow computed as difference: flow = coords1 - coords0
#         return coords0, coords1

#     def upsample_flow(self, flow, mask):
#         """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
#         N, _, H, W = flow.shape
#         mask = mask.view(N, 1, 9, 8, 8, H, W)
#         mask = torch.softmax(mask, dim=2)

#         up_flow = F.unfold(8 * flow, [3,3], padding=1)
#         up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

#         up_flow = torch.sum(mask * up_flow, dim=2)
#         up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
#         return up_flow.reshape(N, 2, 8*H, 8*W)


#     def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):

#         N, _, H, W = image1.shape
#         image1_res = F.interpolate(image1, (518, 518), mode="bilinear", align_corners = False) / 255. 
#         image2_res = F.interpolate(image2, (518, 518), mode="bilinear", align_corners = False) / 255.

#         mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).unsqueeze(0).unsqueeze(2).unsqueeze(2).cuda()
#         std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).unsqueeze(0).unsqueeze(2).unsqueeze(2).cuda()

#         image1_res = image1_res / mean - std 
#         image2_res = image2_res / mean - std

#         im1_path1, depth1 = self.depth_model.forward(image1_res.float())
#         im2_path1, depth2 = self.depth_model.forward(image2_res.float())

#         im1_path1 = F.interpolate(im1_path1, (H, W), mode="bilinear", align_corners = False)
#         im2_path1 = F.interpolate(im2_path1, (H, W), mode="bilinear", align_corners = False)
#         bases1 = self.create_bases(F.interpolate(depth1, (H, W), mode="bilinear", align_corners = False))  
#         bases2 = self.create_bases(F.interpolate(depth2, (H, W), mode="bilinear", align_corners = False))          
        
#         mono1 = self.merge_head(im1_path1)
#         mono2 = self.merge_head(im2_path1)

#         image1 = 2 * (image1 / 255.0) - 1.0
#         image2 = 2 * (image2 / 255.0) - 1.0

#         image1 = image1.contiguous()
#         image2 = image2.contiguous()

#         hdim = self.hidden_dim
#         cdim = self.context_dim

#         padder = InputPadder(image1.shape)
#         image1, image2 = padder.pad(image1, image2)
#         bases1 = padder.pad(bases1)
#         bases2 = padder.pad(bases2)
        
#         N, _, H, W = image1.shape
#         H8, W8 = H // 8, W // 8
#         dilation = torch.ones(N, 1, H//8, W//8, device=image1.device)

#         # run the feature network
#         with torch.amp.autocast('cuda', enabled=self.args.mixed_precision):
#             fmap1, fmap2 = self.fnet([image1, image2])        
#             fmap1 = self.flow_adapter(fmap1)
#             fmap2 = self.flow_adapter(fmap2)

#         mono1_8 = F.interpolate(mono1, (H8, W8), mode="bilinear", align_corners=False)
#         mono2_8 = F.interpolate(mono2, (H8, W8), mode="bilinear", align_corners=False)

#         fmap1 = torch.cat([fmap1, mono1_8], dim=1)
#         fmap2 = torch.cat([fmap2, mono2_8], dim=1)

#         fmap1 = fmap1.float()
#         fmap2 = fmap2.float()

#         if self.args.alternate_corr:
#             corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
#         else:
#             corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

#         # run the context network
#         # cnet_input = torch.cat((image1, image2), 1)
#         with torch.amp.autocast('cuda', enabled=self.args.mixed_precision):
#             cnet = self.cnet(image1)
#         cnet = cnet.float()
#         net, inp = torch.split(cnet, [hdim, cdim], dim=1)


#         bases_feat = self.bnet(bases1)                # [B, hdim+cdim, H8, W8]
#         netbases, inpbases = torch.split(bases_feat, [hdim, cdim], dim=1)

#         net = net + torch.tanh(netbases)
#         inp = inp + torch.relu(inpbases)

#         #run depthhead
#         depth1_hw = F.interpolate(depth1, (H, W), mode="bilinear", align_corners=False)
#         depth2_hw = F.interpolate(depth2, (H, W), mode="bilinear", align_corners=False)

#         depth1_pred1 = self.depthhead(depth1_hw)
#         depth2_pred1 = self.depthhead(depth2_hw)


#         #init flow
#         coords0, coords1 = self.initialize_flow(image1)

#         if flow_init is not None:
#             coords1 = coords1 + flow_init

#         flow_predictions = []

#         for itr in range(iters):
#             coords1 = coords1.detach()
#             # print("feat fmap shape:", fmap1.shape)  # 一般是 [B, C, Hc, Wc]
#             # print("coords1 shape before flatten:", coords1.shape)

#             corr = corr_fn(coords1) # index correlation volume

#             flow = coords1 - coords0
#             with torch.amp.autocast('cuda', enabled=self.args.mixed_precision):
#                 net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

#             # F(t+1) = F(t) + \Delta(t)
#             coords1 = coords1 + delta_flow

#             # upsample predictions
#             if up_mask is None:
#                 flow_up = upflow8(coords1 - coords0)
#             else:
#                 flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
#             flow_predictions.append(flow_up)

#         if test_mode:
#             return coords1 - coords0, flow_up
            
#         return flow_predictions, depth1_pred1, depth2_pred1
