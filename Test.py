import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from utils.dataloader import test_dataset
from utils.eval import Evaluator
from tqdm import tqdm
import time
from lib.DCRNet import DCRNet

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/DCRNet_{}/model.pth')
parser.add_argument('--data_root', type=str, default='/data2/yinzijin/dataset/{}/TestDataset/')
parser.add_argument('--save_root', type=str, default='./results/DCRNet/{}/')
parser.add_argument('--gpu', type=str, default='0', help='GPUs used')

for _data_name in ['piccolo']:
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    data_path = opt.data_root.format(_data_name)
    save_path = opt.save_root.format(_data_name)
    load_path = opt.pth_path.format(_data_name)
    
    model = DCRNet(bank_size = 40)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(load_path))
    model = model.cuda()
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    evaluator = Evaluator()
    
    bar = tqdm(test_loader.images)  
    for i in bar:
        image, gt, name = test_loader.load_data()
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        gt = gt.cuda()
        
        pred = model(image)
        res = pred[len(pred)-1].sigmoid()
        
        evaluator.update(res, gt)
        
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)
    evaluator.show()