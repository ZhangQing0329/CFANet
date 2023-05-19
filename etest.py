import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.cfanet import Net
from utils.tdataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/CFANet.pth')

for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
    data_path = '/home/fiona/Desktop/datasets/COD/{}/'.format(_data_name)
    save_path = './results/CFANet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = Net()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/gt/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, _, _,  = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path+name, (res*255).astype(np.uint8))
        

