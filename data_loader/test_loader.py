# if vis_y_image.shape[2] % 8 != 0:
#     a = vis_y_image.shape[2] // 8
#     b = vis_y_image.shape[3] // 8
#     torch.reshape(vis_y_image, [vis_y_image.shape[0], vis_y_image.shape[1], a * 8, b * 8])
import os
import torch
from PIL import Image,ImageStat
from torch.utils import data
from torchvision import transforms
from model.common import RGB2YCrCb
import numpy as np
import cv2
import time
to_tensor = transforms.Compose([transforms.ToTensor()])


class msrs_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'vis':
                self.vis_path = temp_path  # 获得可见光路径
        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称
        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')  # 获取红外图像
        vis_image = Image.open(os.path.join(self.vis_path, name))

        # if inf_image.size[0] % 8 != 0 or inf_image.size[1] % 8 != 0:
        #     a = inf_image.size[0] // 8
        #     b = inf_image.size[1] // 8
        #     inf_image=inf_image.resize((a*8,b*8),Image.ANTIALIAS)
        #     vis_image=vis_image.resize((a*8,b*8),Image.ANTIALIAS)
        # inf_path = "./test/ir/"
        # vi_path = "./test/vis/"
        # inf_image.save(inf_path+name)
        # vis_image.save(vi_path+name)
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name

    def __len__(self):
        return len(self.name_list)


