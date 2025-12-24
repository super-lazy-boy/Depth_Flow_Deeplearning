# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """
    FlowSeek loss over a sequence of predictions.

    Args:
        output: dict from FlowSeek forward() in training mode.
                Must contain:
                  - output['flow']: list of flow predictions, each [B,2,H,W]
                  - output['nf']:   list of per-pixel loss maps, each [B,1,H,W] or [B,H,W]
        flow_gt: [B,2,H,W] ground-truth optical flow
        valid:   [B,H,W]   valid pixel mask (>=0.5 means valid)
        gamma:   temporal discount factor (later predictions get higher weight)
        max_flow: ignore extremely large motions

    Returns:
        flow_loss: scalar tensor used for backprop
        metrics:  dict for logging (no grad)
    """
    n_predictions = len(output['flow'])
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()                 # [B,H,W]
    if valid.dim() == 4 and valid.size(1) == 1:
        valid = valid[:, 0]   # squeeze channel
    elif valid.dim() == 2:
        valid = valid.unsqueeze(0)

    valid_mask = valid >= 0.5

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        loss_i = output['nf'][i]                                # [B,1,H,W] or [B,H,W]
        if loss_i.dim() == 3:
            loss_i = loss_i.unsqueeze(1)                        # -> [B,1,H,W]

        safe = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach()))
        final_mask = safe & valid_mask[:, None]                 # [B,1,H,W]

        denom = final_mask.sum().clamp_min(1.0)                 # avoid division by zero
        flow_loss = flow_loss + i_weight * ((final_mask * loss_i).sum() / denom)

    # metrics for logging only (do not affect gradients)
    with torch.no_grad():
        flow_final = output['flow'][-1]                         # [B,2,H,W]
        epe_map = torch.sum((flow_final - flow_gt) ** 2, dim=1).sqrt()  # [B,H,W]
        epe_valid = epe_map[valid_mask]
        metrics = {
            "epe": epe_valid.mean().item() if epe_valid.numel() > 0 else 0.0,
            "loss": float(flow_loss.item()),
            "valid_ratio": float(valid_mask.float().mean().item()),
        }

    return flow_loss, metrics