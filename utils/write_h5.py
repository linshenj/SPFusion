# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : wirte_h5.py

Author     ：yujing_rao
"""
import argparse
import glob
import os
import random

import cv2

import numpy as np
import torch
from tqdm import tqdm


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
def check_args(args):
    check_folder(args.data_dir)
    # check_folder(args.result_dir)
    # check_folder(args.summary_dir)
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args
def parse_args():
    desc = "ATFusionGAN implementation of Fusion use GAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--is_train', type=bool, default=True, help='True for training, False for testing [True]')
    parser.add_argument('--sn', type=bool, default=True, help='using spectral norm')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Name of checkpoint directory [checkpoint]')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='Directory name to save the generated images')
    parser.add_argument('--sample_dir', type=str, default='sample',
                        help='Name of sample directory [sample]')
    parser.add_argument('--epoch', type=int, default=30, help='Number of epoch [100]')
    parser.add_argument('--c_dim', type=int, default=1, help='Dimension of image color. [1]')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch images [128]')
    parser.add_argument('--image_size', type=int, default=128, help='he size of image to use [128]')
    parser.add_argument('--label_size', type=int, default=128, help='he size of image to use [128]')
    parser.add_argument('--stride', type=int, default=14, help='The size of stride to apply input image [14]')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='The learning rate of gradient descent algorithm [1e-4]')
    parser.add_argument('--Net', default='result/dis/')
    return check_args(parser.parse_args())

def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1)
    return img[:, :]

def imread_gray(path):
    img = cv2.imread(path)/255
    return img[:, :,0]
def input_setup(config, data_dir):
    # Load data path
        # 取到所有的原始图片的地址
    data = prepare_data(config, dataset=data_dir)
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6
    for i in range(len(data)):
        input_ = (imread(data[i]) - 127.5) / 127.5
        label_ = input_

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        # 按14步长采样小patch
        for x in range(0, h - config.image_size + 1, config.stride):
            for y in range(0, w - config.image_size + 1, config.stride):
                sub_input = input_[int(x):int(x + config.image_size),
                            int(y):int(y + config.image_size)]  # [33 x 33]
                # 注意这里的padding，前向传播时由于卷积是没有padding的，所以实际上预测的是测试patch的中间部分
                sub_label = label_[int(x + padding):int(x + padding + config.label_size),
                            int(y + padding):int(y + padding + config.label_size)]  # [21 x 21]
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]
    return arrdata,arrlabel

def prepare_data(config, dataset):
    data_dir = os.path.join("data", dataset)
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    return data

def train_step(ir_images,ir_labels,vi_images,vi_labels,discriminator,opt_discriminator,net,optimizer_net,d_loss_ob,tq,epoch):
        d_loss=0
        fused_img,_,_= net(ir_images, vi_images)
        with torch.no_grad():
            fusion_out=fused_img
        for d_train_number in range(2):
            real_ir=ir_labels
            real_vi=vi_labels
            opt_discriminator.zero_grad()
            real_ir_pro=discriminator(real_ir)
            real_vi_pro=discriminator(real_vi)
            fake_fused_pro=discriminator(fusion_out)
            d1_loss=d_loss_ob(real_vi_pro,1,0)+d_loss_ob(real_ir_pro,0,1)
            d2_loss=d_loss_ob(fake_fused_pro,0,0)
            d_train_loss=d1_loss+d2_loss
            d_loss+=d_train_loss.cpu().item()
            d_train_loss.backward(retain_graph=True)
            opt_discriminator.step()
            tq.set_postfix(epoch=epoch,loss_d = d_train_loss.item())
        optimizer_net.zero_grad()

def lab():
    config = parse_args()
    print("标签制作开始")
    train_data_ir, train_label_ir=input_setup(config, "ir")
    train_data_vi, train_label_vi=input_setup(config, "vis")
    print("标签制作完成")
    return train_data_ir, train_label_ir,train_data_vi, train_label_vi


def main(train_data_ir, train_label_ir,train_data_vi, train_label_vi,dis,opt_discriminator,net,optimizer_net,d_loss_ob,epoch):
        dis.train()
        config = parse_args()
        batch_size = config.batch_size
        print("鉴别器训练开始")
        batch_idxs = len(train_data_ir) // config.batch_size
        tq = tqdm(range(1, 1 + batch_idxs), total=len(range(1, 1 + batch_idxs)))
        for idx in tq:
            start_idx = (idx - 1) * batch_size
            ir_images = train_data_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
            ir_labels = train_label_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
            vi_images = train_data_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
            vi_labels = train_label_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])

            ir_images = torch.tensor(ir_images).float().cuda()
            ir_labels = torch.tensor(ir_labels).float().cuda()
            vi_images = torch.tensor(vi_images).float().cuda()
            vi_labels = torch.tensor(vi_labels).float().cuda()
            # return g_loss(all)，g1_train_loss
            train_step(ir_images, ir_labels, vi_images, vi_labels,dis,opt_discriminator,net,optimizer_net,d_loss_ob,tq,epoch)
        torch.save(dis.state_dict(), f'{config.Net}/{epoch}.pth')