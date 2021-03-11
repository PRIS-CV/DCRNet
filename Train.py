import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.loss import BCEDiceLoss
from utils.dataloader import get_loader
from utils.utils import AvgMeter
import torch.nn.functional as F
from lib.DCRNet import DCRNet

def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):           
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts = pack['image'], pack['label']
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        # ---- forward ----
        pred = model(images)
        # ---- loss function ----
        loss4 = BCEDiceLoss(pred[4], gts)
        loss3 = BCEDiceLoss(pred[3], gts)
        loss2 = BCEDiceLoss(pred[2], gts)
        loss1 = BCEDiceLoss(pred[1], gts)
        loss0 = BCEDiceLoss(pred[0], gts)
        loss = loss0 + loss1 + loss2 + loss3 + loss4
        # ---- backward ----
        loss.backward()
        optimizer.step()
        # ---- recording loss ----
        loss_record.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[loss: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))
    save_path = './snapshots/{}/'.format(opt.save_root)
    os.makedirs(save_path, exist_ok=True)
    if epoch % 50 == 0:
        torch.save(model.state_dict(), save_path + 'model-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'model-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=224, help='training dataset size')
    parser.add_argument('--train_path', type=str,
                        default='/data2/yinzijin/dataset/EndoScene/TrainDataset/', help='path to train dataset')
    parser.add_argument('--save_root', type=str,
                        default='DCRNet_EndoScene')
    parser.add_argument('--gpu', type=str,
                        default='0', help='GPUs used')
    opt = parser.parse_args()

    # ---- build models ----
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    model = DCRNet()
    model = nn.DataParallel(model).cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)
    
    for epoch in range(1, opt.epoch+1):
        train(train_loader, model, optimizer, epoch)


