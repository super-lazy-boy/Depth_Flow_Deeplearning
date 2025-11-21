import os
import torch
from torch import nn

dinov3_dir = os.path.join(".", "dinov3")
dinov3_weights_dir = os.path.join(".", "checkpoints")

class dinov3_Backbone(nn.Module):
    def __init__(self,
                 model=None,
                 ):
        super(dinov3_Backbone, self).__init__()
        if model == "vitb16":
            hub_model_name = "dinov3_vitb16"
            weights = "dinov3_vitb16_pretrain.pth"
        elif model == "vitl16":
            hub_model_name = "dinov3_vitl16"
            weights = "dinov3_vitl16_pretrain.pth"
        elif model == "vith16plus":
            hub_model_name = "dinov3_vith16plus"
            weights = "dinov3_vith16plus_pretrain.pth"
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        weights_path = os.path.join(dinov3_weights_dir, weights)

        self.model = torch.hub.load(
            dinov3_dir,
            hub_model_name,
            source="local",
            weights=weights_path,
        )
        for p in self.model.parameters():
                p.requires_grad = False
        
        self.model.eval()

    def forward(self,x):
        out = self.model.forward_features(x)  

        # shape: [B, N, C]
        tokens = out["x_norm_patchtokens"]    
        B, N, C = tokens.shape

        H_feat = int(N ** 0.5)
        W_feat = H_feat
        tokens = tokens.permute(0, 2, 1)          # [B, C, N]
        feat = tokens.reshape(B, C, H_feat, W_feat)  # [B, C, H_feat, W_feat]

        return feat

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
