import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


DEPTH_ANYTHING_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../Depth-Anything-V2")
)
sys.path.insert(0, DEPTH_ANYTHING_ROOT)
from depth_anything_v2.dpt import DepthAnythingV2

ckpt_path = "Depth_Flow_Deeplearning/checkpoints/depth_anything_v2_vitb.pth"

class DepthAnythingV2Frozen(nn.Module):
    def __init__(self, args, encoder='vitb'):
        super(DepthAnythingV2Frozen, self).__init__()
        self.args = args

        # Initialize DepthAnyThingV2 model
        self.depth_model = DepthAnythingV2(encoder=encoder )
        self.depth_model.load_state_dict(torch.load(ckpt_path), strict=False)
        self.depth_model.eval()  # Set to evaluation mode
        for param in self.depth_model.parameters():
            param.requires_grad = False  # Freeze parameters

        self._hook = self.net.depth_head.scratch.output_conv1.register_forward_hook(self._save_feat)

    def forward(self, x):
        depth_outputs = self.depth_model(x)
        return depth_outputs