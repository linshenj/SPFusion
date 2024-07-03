import torch
import torchprofile
from torch import nn
from kornia.filters import SpatialGradient
from torch import Tensor


class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class EdgeDetect(nn.Module):
    def __init__(self):
        super(EdgeDetect, self).__init__()
        self.spatial = SpatialGradient('diff')
        self.max_pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        y = self.max_pool(u)
        return y


class ConvConv(nn.Module):
    def __init__(self, a_channels, b_channels, c_channels):
        super(ConvConv, self).__init__()

        self.conv_1 = nn.Conv2d(a_channels, b_channels, (3, 3), padding=(1, 1))
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(b_channels, c_channels, (3, 3), padding=(2, 2), dilation=(2, 2))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        return x

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv_1 = ConvConv(1, 32, 32)
        self.conv_2 = ConvConv(32, 64, 128)
        self.conv_3 = ConvConv(128, 64, 32)
        self.conv_4 = nn.Conv2d(32, 1, (1, 1))
        self.ed = EdgeDetect()

    def forward(self, x):
        # edge detect
        e = self.ed(x)
        x = x + e
        # attention
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x


class MCMDAF(nn.Module):
    def __init__(self):
        super(MCMDAF, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.gap_channel = nn.AdaptiveAvgPool2d(1)
        self.gap_spatial = nn.AdaptiveAvgPool2d(1)

    def forward(self, vi_feature, ir_feature):
        batch_size, channels, _, _ = vi_feature.size()

        # 通道互补特征计算
        sub_vi_ir = vi_feature - ir_feature
        vi_ir_channel_attention = self.sigmoid(self.gap_channel(sub_vi_ir))
        vi_ir_channel_div = sub_vi_ir * self.relu(vi_ir_channel_attention)

        sub_ir_vi = ir_feature - vi_feature
        ir_vi_channel_attention = self.sigmoid(self.gap_channel(sub_ir_vi))
        ir_vi_channel_div = sub_ir_vi * self.relu(ir_vi_channel_attention)

        # 空间互补特征计算
        vi_ir_spatial_attention = self.sigmoid(self.gap_spatial(sub_vi_ir))
        vi_ir_spatial_div = sub_vi_ir * self.relu(vi_ir_spatial_attention)

        ir_vi_spatial_attention = self.sigmoid(self.gap_spatial(sub_ir_vi))
        ir_vi_spatial_div = sub_ir_vi * self.relu(ir_vi_spatial_attention)

        # 整合通道互补特征
        vi_feature_channel = vi_feature + ir_vi_channel_div
        ir_feature_channel = ir_feature + vi_ir_channel_div

        # 整合空间互补特征
        vi_feature_spatial = vi_feature + ir_vi_spatial_div
        ir_feature_spatial = ir_feature + vi_ir_spatial_div

        # 将通道互补特征和空间互补特征相乘得到最终互补特征
        vi_feature_final = vi_feature_channel * vi_feature_spatial
        ir_feature_final = ir_feature_channel * ir_feature_spatial

        return vi_feature_final, ir_feature_final


def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)

        self.vi_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

        self.mcmdaf1 = MCMDAF()
        self.mcmdaf2 = MCMDAF()
        self.mcmdaf3 = MCMDAF()

        self.conv_vi = reflect_conv(in_channels=1, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv_ir = reflect_conv(in_channels=1, kernel_size=3, out_channels=128, stride=1, pad=1)

        self.atn1 = Attention()
        self.atn2 = Attention()

        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        rate_vi = self.atn1(y_vi_image)
        rate_ir = self.atn2(ir_image)
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))
        vi_out, ir_out = self.mcmdaf1(activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out)))
        vi_out, ir_out = self.mcmdaf2(activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out)))
        vi_out, ir_out = self.mcmdaf3(activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out)))
        vi_out, ir_out = activate(self.vi_conv5(vi_out)), activate(self.ir_conv5(ir_out))
        vi_out = self.softmax(rate_vi) * vi_out
        ir_out = self.softmax(rate_ir) * (ir_out+ir_out)
        return vi_out, ir_out

class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.conv3 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=1, stride=1, padding=0)
    def forward(self, x):
        activate = nn.LeakyReLU()
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = reflect_conv(in_channels=256, kernel_size=3, out_channels=256, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=256, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.LeakyReLU()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, y_vi_image, ir_image):
        vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)
        encoder_out = Fusion(vi_encoder_out, ir_encoder_out)
        fused_image = self.decoder(encoder_out)
        return fused_image,vi_encoder_out,ir_encoder_out



if __name__ == '__main__':
    img1 = torch.randn((1,1,128,128)).cuda()
    img2 = torch.randn((1,1,128,128)).cuda()
    model = Net().cuda()
    flops = torchprofile.profile_macs(model, (img1,img2))
    print(f"FLOPs: {flops}")