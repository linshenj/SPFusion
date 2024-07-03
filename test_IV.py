"""测试融合网络"""
import argparse
import os
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data_loader.test_loader import msrs_data
from model.common import YCrCb2RGB, clamp
from model.netk import Net



def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MT-fuse')
    parser.add_argument('--dataset_path', metavar='DIR', default='F:/A/Source-Image/S',
                        help='path to dataset (default: imagenet)')  # 测试数据存放位置
    parser.add_argument('--save_path', default='output')  # 融合结果存放位置
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # parser.add_argument('--Net', default='./pretrain/net/the best.pth')
    parser.add_argument('--Net', default='./pretrain/net/the best.pth')
    # parser.add_argument('--Net', default='./result/net/1.pth')

    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)
    test_dataset = msrs_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    #######加载模型
    net = Net().cuda()

    #########推理
    net.eval()

    ###############加载
    net.load_state_dict(torch.load(args.Net))
    para = sum([np.prod(list(p.size())) for p in net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(net._get_name(), para * type_size / 1000 / 1000))

    total = sum([param.nelement() for param in net.parameters()])
    print('Number	of	parameter: {:4f}M'.format(total / 1e6))
    fuse_time = []
    ##########加载数据
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:
            vis_y_image = vis_y_image.cuda()
            cb = cb.cuda()
            cr = cr.cuda()
            inf_image = inf_image.cuda()
            start = time.time()
            #########编码
            fused,_,_ = net(vis_y_image,inf_image)
            end = time.time()
            fuse_time.append(end - start)
            ###########转为rgb
            fused = clamp(fused)

            rgb_fused_image = YCrCb2RGB(fused[0], cb[0], cr[0])
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
            rgb_fused_image.save(f'{args.save_path}/{name[0]}')
    mean = statistics.mean(fuse_time[1:])
    print(f'fuse avg time: {mean:.4f}')



