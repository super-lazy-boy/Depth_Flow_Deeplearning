import torch
from torch import nn, functional as F
import numpy as np

dinov3_dir = "..\dinov3"
dinov3_weights_dir = "..\checkpoints"

class dinov3_Backbone(nn.Module):
    def __init__(self,
                 model=None,
                 ):
        super(dinov3_Backbone, self).__init__()
        if model == "vitb16":
            hub_model_name = "dinov3_vitb16"
            default_weights = "dinov3_vitb16_pretrain.pth"
        elif model == "vitl16":
            hub_model_name = "dinov3_vitl16"
            default_weights = "dinov3_vitl16_pretrain.pth"
        elif model == "vith16plus":
            hub_model_name = "dinov3_vith16plus"
            default_weights = "dinov3_vith16plus_pretrain.pth"
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        weights_name = default_weights
        weights_path = f"{dinov3_weights_dir}/{weights_name}"

        self.model = torch.hub.load(
            dinov3_dir,
            hub_model_name,
            source="local",
            weights=weights_path,
        )

    def forward(self,x):
        self.model.eval(requires_grad=False)
        
        out = self.model.forward_features(x)  
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 假设输入 512x512
    dummy = torch.randn(2, 3, 512, 512).to(device)

    backbone = dinov3_Backbone(
        model="vitb16"
    ).to(device)

    with torch.no_grad():
        feat = backbone(dummy)

    print("Input shape :", dummy.shape)  # [2, 3, 512, 512]
    print("Feature shape:", feat.shape)  
