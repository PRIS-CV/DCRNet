import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image
import argparse
from tqdm import tqdm
import math

class Evaluator:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.MAE = list()
        self.Recall = list()
        self.Precision = list()
        self.Accuracy = list()
        self.Dice = list()       
        self.IoU_polyp = list()
        self.boundary_F = list()
        self.S_measure = list()
        

    def seg2bmap(self, seg, width=None, height=None):
        """
        From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries.  The boundary pixels are offset by 1/2 pixel towards the
        origin from the actual segment boundary.
        Arguments:
            seg     : Segments labeled from 1..k.
            width	  :	Width of desired bmap  <= seg.shape[1]
            height  :	Height of desired bmap <= seg.shape[0]
        Returns:
            bmap (ndarray):	Binary boundary map.
         David Martin <dmartin@eecs.berkeley.edu>
         January 2003
         """
        seg = seg.astype(np.bool)
        seg[seg > 0] = 1

        assert np.atleast_3d(seg).shape[2] == 1

        width = seg.shape[1] if width is None else width
        height = seg.shape[0] if height is None else height

        h, w = seg.shape[:2]

        ar1 = float(width) / float(height)
        ar2 = float(w) / float(h)

        assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
            'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

        e = np.zeros_like(seg)
        s = np.zeros_like(seg)
        se = np.zeros_like(seg)

        e[:, :-1] = seg[:, 1:]
        s[:-1, :] = seg[1:, :]
        se[:-1, :-1] = seg[1:, 1:]

        b = seg ^ e | seg ^ s | seg ^ se
        b[-1, :] = seg[-1, :] ^ e[-1, :]
        b[:, -1] = seg[:, -1] ^ s[:, -1]
        b[-1, -1] = 0

        if w == width and h == height:
            bmap = b
        else:
            bmap = np.zeros((height, width))
            for x in range(w):
                for y in range(h):
                    if b[y, x]:
                        j = 1 + math.floor((y - 1) + height / h)
                        i = 1 + math.floor((x - 1) + width / h)
                        bmap[j, i] = 1

        return bmap

    def eval_boundary(self, foreground_mask, gt_mask, bound_th=0.008):
        """
        Compute mean,recall and decay from per-frame evaluation.
        Calculates precision/recall for boundaries between foreground_mask and
        gt_mask using morphological operators to speed it up.
        Arguments:
            foreground_mask (ndarray): binary segmentation image.
            gt_mask         (ndarray): binary annotated image.
        Returns:
            F (float): boundaries F-measure
            P (float): boundaries precision
            R (float): boundaries recall
        """
        foreground_mask = foreground_mask.data.cpu().numpy().squeeze()
        foreground_mask = (foreground_mask >= 0.5)
        gt_mask = gt_mask.data.cpu().numpy().squeeze()
        gt_mask = (gt_mask >= 0.5)
        assert np.atleast_3d(foreground_mask).shape[2] == 1

        bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

        # Get the pixel boundaries of both masks
        fg_boundary = self.seg2bmap(foreground_mask);

        gt_boundary = self.seg2bmap(gt_mask);

        from skimage.morphology import binary_dilation, disk

        fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
        gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

        # Get the intersection
        gt_match = gt_boundary * fg_dil
        fg_match = fg_boundary * gt_dil

        # Area of the intersection
        n_fg = np.sum(fg_boundary)
        n_gt = np.sum(gt_boundary)

        # % Compute precision and recall
        if n_fg == 0 and n_gt > 0:
            precision = 1
            recall = 0
        elif n_fg > 0 and n_gt == 0:
            precision = 0
            recall = 1
        elif n_fg == 0 and n_gt == 0:
            precision = 1
            recall = 1
        else:
            precision = np.sum(fg_match) / float(n_fg)
            recall = np.sum(gt_match) / float(n_gt)

        # Compute F measure
        if precision + recall == 0:
            F = 0
        else:
            F = 2 * precision * recall / (precision + recall);

        return F

    def evaluate(self, pred, gt):
        
        pred_binary = (pred >= 0.5).float().cuda()
        pred_binary_inverse = (pred_binary == 0).float().cuda()

        gt_binary = (gt >= 0.5).float().cuda()
        gt_binary_inverse = (gt_binary == 0).float().cuda()
        
        MAE = torch.abs(pred_binary - gt_binary).mean().cuda(0)
        TP = pred_binary.mul(gt_binary).sum().cuda(0)
        FP = pred_binary.mul(gt_binary_inverse).sum().cuda(0)
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda(0)
        FN = pred_binary_inverse.mul(gt_binary).sum().cuda(0)

        if TP.item() == 0:
            TP = torch.Tensor([1]).cuda(0)
        # recall
        Recall = TP / (TP + FN)
        # Precision or positive predictive value
        Precision = TP / (TP + FP)
        # F1 score = Dice
        Dice = 2 * Precision * Recall / (Precision + Recall)
        # Overall accuracy
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        # IoU for poly
        IoU_polyp = TP / (TP + FP + FN)

        return MAE.data.cpu().numpy().squeeze(), Recall.data.cpu().numpy().squeeze(), Precision.data.cpu().numpy().squeeze(), Accuracy.data.cpu().numpy().squeeze(), Dice.data.cpu().numpy().squeeze(), IoU_polyp.data.cpu().numpy().squeeze()
    
    def Eval_Smeasure(self, pred, gt, alpha=0.5):
        y = gt.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            gt[gt>=0.5] = 1
            gt[gt<0.5] = 0
            Q = alpha * self._S_object(pred, gt) + (1-alpha) * self._S_region(pred, gt)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        return Q.item()
        
    def update(self, pred, gt):
        mae, recall, precision, accuracy, dice, ioU_polyp = self.evaluate(pred, gt)
        boundary_F = self.eval_boundary(pred, gt)
        s_measure = self.Eval_Smeasure(pred, gt)
        
        self.MAE.append(mae)
        self.Recall.append(recall)
        self.Precision.append(precision)
        self.Accuracy.append(accuracy)
        self.Dice.append(dice)       
        self.IoU_polyp.append(ioU_polyp)
        self.boundary_F.append(boundary_F)
        self.S_measure.append(s_measure)

    def show(self):
        print("\n ===============Metrics===============\n")
        print("MAE : " + str(np.mean(self.MAE)))
        print("Recall : " + str(np.mean(self.Recall)))
        print("Precision : " + str(np.mean(self.Precision)))
        print("Accuracy : " + str(np.mean(self.Accuracy)))
        print("Dice : " + str(np.mean(self.Dice)))
        print("IoU_polyp : " + str(np.mean(self.IoU_polyp)))
        print("Boundary_F : " + str(np.mean(self.boundary_F)))
        print("S_measure : " + str(np.mean(self.S_measure)))
        
    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1-pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        
        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        # print(Q)
        return Q
    
    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0,cols)).cuda().float()
                j = torch.from_numpy(np.arange(0,rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0,cols)).float()
                j = torch.from_numpy(np.arange(0,rows)).float()
            X = torch.round((gt.sum(dim=0)*i).sum() / total)
            Y = torch.round((gt.sum(dim=1)*j).sum() / total)
        return X.long(), Y.long()
    
    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h*w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
        
        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q