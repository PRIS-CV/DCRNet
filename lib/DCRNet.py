import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from .UNet_ResNet34 import ResNet34Unet
from .modules import *
   
class DCRNet(ResNet34Unet):
    def __init__(self,
                 bank_size=20,
                 num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True,
                 feat_channels=512
                 ):
        super().__init__(num_classes=1,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True)
        
        self.bank_size = bank_size
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))  # memory bank pointer
        self.register_buffer("bank", torch.zeros(self.bank_size, feat_channels, num_classes))  # memory bank
        self.bank_full = False
        
        # =====Attentive Cross Image Interaction==== #
        self.feat_channels = feat_channels
        self.L = nn.Conv2d(feat_channels, num_classes, 1)
        self.X = conv2d(feat_channels, 512, 3)
        self.phi = conv1d(512, 256)
        self.psi = conv1d(512, 256)
        self.delta = conv1d(512, 256)
        self.rho = conv1d(256, 512)
        self.g = conv2d(512 + 512, 512, 1)
        # =========Dual Attention========== #
        self.sa_head = PAM_Module(feat_channels)
        #=========Attention Fusion=========#
        self.fusion = nn.Conv2d(feat_channels, feat_channels, 1)
    #==Initiate the pointer of bank buffer==#
    def init(self):
        self.bank_ptr[0] = 0
        self.bank_full = False
        
    @torch.no_grad() #这句很重要！！！！
    def update_bank(self, x):
        ptr = int(self.bank_ptr)
        batch_size = x.shape[0]
        vacancy = self.bank_size - ptr
        if batch_size >= vacancy:
            self.bank_full = True
        pos = min(batch_size, vacancy)
        self.bank[ptr:ptr+pos] = x[0:pos].clone()
        # update pointer
        ptr = (ptr + pos) % self.bank_size
        self.bank_ptr[0] = ptr
        
    def down(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)        
        return e4, e3, e2, e1
    
    def up(self, feat, e3, e2, e1, x):
        center = self.center(feat)
        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
 
        f1 = self.finalconv1(d1)
        f2 = self.finalconv2(d2)
        f3 = self.finalconv3(d3)
        f4 = self.finalconv4(d4)
                
        f4 = F.interpolate(f4, scale_factor=8, mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        
        return f4, f3, f2, f1
   
    def region_representation(self, input):
        X = self.X(input)
        L = self.L(input)
        aux_out = L
        batch, n_class, height, width = L.shape
        l_flat = L.view(batch, n_class, -1)
        # M = B * N * HW
        M = torch.softmax(l_flat, -1)
        channel = X.shape[1]
        # X_flat = B * C * HW
        X_flat = X.view(batch, channel, -1)
        # f_k = B * C * N
        f_k = (M @ X_flat.transpose(1, 2)).transpose(1, 2)
        return aux_out, f_k, X_flat, X
    
    def attentive_interaction(self, bank, X_flat, X):
        batch, n_class, height, width = X.shape
        # query = S * C
        query = self.phi(bank).squeeze(dim=2)
        # key: = B * C * HW
        key = self.psi(X_flat)
        # logit = HW * S * B (cross image relation)
        logit = torch.matmul(query, key).transpose(0,2)
        # attn = HW * S * B
        attn = torch.softmax(logit, 2) ##softmax维度要正确
        
        # delta = S * C
        delta = self.delta(bank).squeeze(dim=2)
        # attn_sum = B * C * HW
        attn_sum = torch.matmul(attn.transpose(1,2), delta).transpose(1,2)
        # x_obj = B * C * H * W
        X_obj = self.rho(attn_sum).view(batch, -1, height, width)

        concat = torch.cat([X, X_obj], 1)
        out = self.g(concat)
        return out
            
    def forward(self, x, flag='train'):
        batch_size = x.shape[0]
        #=== Stem ===#
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
 
        #=== Encoder ===#
        e4, e3, e2, e1  = self.down(x_)        
        #=== Attentive Cross Image Interaction ===#
        aux_out, patch, feats_flat, feats = self.region_representation(e4)
        if flag == 'train':
            self.update_bank(patch)
            ptr = int(self.bank_ptr)
            if self.bank_full == True:
                feature_aug = self.attentive_interaction(self.bank, feats_flat, feats)
            else:
                feature_aug = self.attentive_interaction(self.bank[0:ptr], feats_flat, feats)
        elif flag == 'test':
            feature_aug = self.attentive_interaction(patch, feats_flat, feats)
        #=== Dual Attention ===#
        sa_feat = self.sa_head(e4)
        #=== Fusion ===#
        feats = sa_feat + feature_aug
        #=== Decoder ===#
        f4, f3, f2, f1 = self.up(feats, e3, e2, e1, x)
        aux_out = F.interpolate(aux_out, scale_factor=32, mode='bilinear', align_corners=True)
        return aux_out, f4, f3, f2, f1
        
    