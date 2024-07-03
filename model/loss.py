import cv2
from torch import nn
from torchvision.models import vgg19
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torchvision.transforms import transforms
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        # 这里不行就采用expend_dims
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)

def constrain_loss_cosine_fixed(out_1, out_2, batch_size, temperature):
    # 计算所有特征向量之间的余弦相似度
    out = torch.cat([out_1, out_2], dim=0)
    out = F.normalize(out, p=2, dim=1)
    sim_matrix = torch.mm(out, out.t().contiguous()) / temperature

    # 创建掩码以排除自比较的相似度，并且避免直接使用自相似度作为正样本对
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix_masked = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # 对sim_matrix进行softmax操作，以便转换为概率分布
    sim_matrix_softmax = F.softmax(sim_matrix_masked, dim=1)

    # 计算正样本对的索引
    # 对于每个样本，其正样本对应在拼接后的输出中的位置
    pos_indices = torch.arange(batch_size, dtype=torch.long, device=sim_matrix.device)
    pos_indices = torch.cat([pos_indices + batch_size, pos_indices], dim=0)  # 调整索引以匹配正样本位置

    # 计算正样本对的概率
    pos_probs = sim_matrix_softmax[torch.arange(2*batch_size, device=sim_matrix.device), pos_indices]

    # 计算损失
    loss = -torch.log(pos_probs).mean()
    return loss

class vgg_loss(torch.nn.Module):
    def __init__(self):
        super(vgg_loss, self).__init__()
        vgg_model = vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:].cuda()
        vgg_model.eval()
        for param in vgg_model.parameters():
            param.requires_grad = False  # 使得之后计算梯度时不进行反向传播及权重更新
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }
        # self.weight = [1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
        self.weight = [1.0, 1.0, 1.0, 1.0, 1.0]

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            # print("vgg_layers name:",name,module)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        #print(output.keys())
        return list(output.values())

    def forward(self, output, gt):
        loss = []
        ###########这里的目的是将输入张量处理为channel=3
        output_ = output
        output = torch.cat((output_, output_), dim=1)
        output = torch.cat((output_, output), dim=1)
        #####################
        gt_ = gt
        gt = torch.cat((gt_, gt_), dim=1)
        gt = torch.cat((gt_, gt), dim=1)
        ###############
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter, (dehaze_feature, gt_feature, loss_weight) in enumerate(
                zip(output_features, gt_features, self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature) * loss_weight)
        return sum(loss)  # /len(loss)


####tv loss 亮度平滑损失

class smoth(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(smoth,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

#三通道一致性损失
class color(torch.nn.Module):
    def __init__(self):
        super(color, self).__init__()

    def forward(self, fused ,cb,cr):
        Drg = torch.pow(cb-cr,2)
        Drb = torch.pow(fused-cr,2)
        Dgb = torch.pow(fused-cb,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k



def contrastive_loss(features_ir, features_vis, temperature=0.5):
    """
    Calculate the contrastive loss between infrared and visible features.

    Args:
    - features_ir (torch.Tensor): Batch of features from infrared images.
    - features_vis (torch.Tensor): Batch of features from visible images.
    - temperature (float): Temperature parameter for scaling the logits.

    Returns:
    - torch.Tensor: Calculated contrastive loss.
    """

    # Normalize features to get unit vectors
    features_ir = F.normalize(features_ir, p=2, dim=1)
    features_vis = F.normalize(features_vis, p=2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(features_ir, features_vis.T) / temperature

    # Labels for positive and negative pairs
    labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)

    # Loss for infrared as anchor
    loss_ir = F.cross_entropy(sim_matrix, labels)

    # Loss for visible as anchor
    loss_vis = F.cross_entropy(sim_matrix.T, labels)

    # Total loss is the average of both losses
    total_loss = (loss_ir + loss_vis) / 2

    return total_loss


def augmented_contrastive_loss(features_ir, features_vis, features_vis_aug, temperature=0.5):
    """
    Compute contrastive loss including augmented visible features.

    Args:
    - features_ir (torch.Tensor): Batch of features from infrared images.
    - features_vis (torch.Tensor): Batch of features from visible images.
    - features_vis_aug (torch.Tensor): Batch of features from augmented visible images.
    - temperature (float): Temperature parameter for scaling the logits.

    Returns:
    - torch.Tensor: Calculated augmented contrastive loss.
    """
    # Calculate basic contrastive loss
    basic_loss = contrastive_loss(features_ir, features_vis, temperature)

    # Normalize augmented visible features
    features_vis_aug = F.normalize(features_vis_aug, p=2, dim=1)

    # Compute similarity matrix for augmented data
    sim_matrix_aug = torch.matmul(features_ir, features_vis_aug.T) / temperature

    # Labels for positive and negative pairs with augmented data
    labels_aug = torch.arange(sim_matrix_aug.size(0)).to(sim_matrix_aug.device)

    # Loss for infrared as anchor against augmented visible
    loss_ir_aug = F.cross_entropy(sim_matrix_aug, labels_aug)

    # Loss for augmented visible as anchor
    loss_vis_aug = F.cross_entropy(sim_matrix_aug.T, labels_aug)

    # Total augmented loss is the average of basic and augmented losses
    total_augmented_loss = (basic_loss + (loss_ir_aug + loss_vis_aug) / 2) / 2

    return total_augmented_loss
#对抗性损失
class Adv_loss(torch.nn.Module):
    def __init__(self):
        super(Adv_loss,self).__init__()
    def forward(self,real,fake):
        D_loss_real = torch.mean(-torch.log(real+0.0000001))
        D_loss_fake = torch.mean(-torch.log(1.- fake +0.0000001))
        D_loss = D_loss_fake+D_loss_real
        return D_loss

#对比度损失
def contrast(x):
    with torch.no_grad():
        mean_x = torch.mean(x, dim=[2, 3], keepdim=True)
        c = torch.sqrt(torch.mean((x - mean_x) ** 2, dim=[2, 3], keepdim=True))
    return c
def contrast_loss(vi,ir,fuse):
    contrast_loss=torch.mean(torch.abs(contrast(fuse))-torch.max(contrast(vi),contrast(ir)))
    return contrast_loss

#分割语义损失
class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

def corr_loss(image_ir, img_vis, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss


def load_and_transform_image(image_tensor):
    # Load the image and compute saliency map
    image_np = image_tensor.cpu().numpy()

    # Convert from PyTorch's channel-first format (C, H, W) to OpenCV's channel-last format (H, W, C)
    if image_tensor.ndim == 3:  # If it's a single image, not a batch
        image_np = np.transpose(image_np, (1, 2, 0))

    # Handle grayscale (2D) and RGB images (3D)
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        # Convert RGB to BGR for OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[2] == 1):
        # If the image is grayscale, ensure it's a 2D array
        image_np = image_np.squeeze()
    else:
        raise ValueError("Unsupported image shape: {}".format(image_np.shape))

    # Compute saliency
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliencyMap = saliency.computeSaliency(image_np.astype('float32'))
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # Apply threshold
    _, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Convert back to PyTorch tensor
    img_tensor = transforms.ToTensor()(threshMap)

    return img_tensor
class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)
def EN_function(image):
    # 计算图像的直方图
    histogram, bins = np.histogram(image, bins=256, range=(0, 255))
    # 将直方图归一化
    histogram = histogram / float(np.sum(histogram))
    # 计算熵
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy



def weight(vi,ir,q_vi,q_ir):
    Q_vi = EN_function(vi)*q_vi
    Q_ir = EN_function(ir)*q_ir
    w = Q_vi * Q_vi/(Q_vi * Q_vi+Q_ir*Q_ir)
    return w

class DLoss(nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()
    def forward(self,image_pro,pos_0,pos_1):
        loss=torch.mean(torch.square(image_pro[:,0]-pos_0))+torch.mean(torch.square(image_pro[:,1]-pos_1))
        return loss

def Seg_loss(pred, label, criteria=None):
    # 计算语义损失
    lb = torch.squeeze(label, 1)
    seg_loss = criteria(pred, lb)
    return seg_loss


if __name__ == '__main__':
    gt = torch.randn(1,1,224,224)
    gt1 = torch.randn(1,1,224,224)
    print(weight(gt,gt1))




