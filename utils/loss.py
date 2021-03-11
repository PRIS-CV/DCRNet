import torch
import torch.nn as nn
import torch.nn.functional as F

def BCEDiceLoss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)   
    smooth = 1
    size = pred.size(0)
    pred_flat = pred.view(size, -1)
    mask_flat = mask.view(size, -1)
    intersection = pred_flat * mask_flat
    dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + mask_flat.sum(1) + smooth)
    dice_loss = 1 - dice_score.sum()/size
    
    return (wbce + dice_loss).mean()