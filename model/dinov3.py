import os
import torch
from torch import nn
from torch.nn import functional as F

dinov3_dir = os.path.join(".", "dinov3")
dinov3_weights_dir = os.path.join(".", "checkpoints")

class dinov3(nn.Module):
    def __init__(self,
                 model=None,
                 ):
        super(dinov3, self).__init__()
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
        B, _, H, W = x.shape

        out = self.model.forward_features(x)  

        # shape: [B, N, C]
        tokens = out["x_norm_patchtokens"]    
        _, N, C = tokens.shape

        patch_size = 16  # ViT-16

        H_feat = H // patch_size
        W_feat = W // patch_size

        tokens = tokens.permute(0, 2, 1).contiguous()          # [B, C, N]
        feat = tokens.reshape(B, C, H_feat, W_feat)  # [B, C, H_feat, W_feat]

        return feat

class Dinov3Encoder(nn.Module):
    def __init__(self, output_dim=128, model='vitb16', dropout=0.0):
        super(Dinov3Encoder, self).__init__()
        self.backbone = dinov3(model=model)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        if model == 'vitb16':
            input_dim = 768
        elif model == 'vitl16':
            input_dim = 1024
        else: 
            ValueError(f"Unsupported model: {model}")
        
        self.conv2 = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.backbone(x)
        x = F.interpolate(x, scale_factor=2.0,
                          mode='bilinear', align_corners=False)#上采样到 1/8 尺度: [B, C_in, H/8, W/8]
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 假设输入 512x512
    dummy = torch.randn(2, 3, 512, 512).to(device)

    backbone = Dinov3Encoder(
        model="vitb16",
        output_dim=128,
        dropout=0.0,
    ).to(device)

    with torch.no_grad():
        feat = backbone(dummy)

    print("Input shape :", dummy.shape)  # [2, 3, 512, 512]
    print("Feature shape:", feat.shape)  
