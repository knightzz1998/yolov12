# -------------------------------------------------------------------#
#    Author       : 章鑫
#    Date         : 2023-10-11 20:22:56
#    LastEditTime : 2023-10-12 17:02:50
#    Description  : RFAConv系列
# -------------------------------------------------------------------#
import logging

import torch
from torch import nn
from einops import rearrange
#RAFConv: Innovating Spatial Attention and Standard Convolutional Operation
#https://arxiv.org/pdf/2304.03198

__all__ = ("SPConv_3x3")

from ultralytics.utils import set_logging

# 日志
set_logging(name="RFCAConv", verbose=True)
logger = logging.getLogger("RFCAConv")

__all__ = ('RFCAConv')

class RFAConv(nn.Module):  # 基于Group Conv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        # 打印输入
        logger.info(f"RFAConv ==> 输入张量形状: {x.shape}")
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        out = self.conv(conv_data)
        # 打印输出
        logger.info(f"RFAConv ==> 输出张量形状: {out.shape}")
        return out


# class RFAConv(nn.Module):  # 基于Unfold实现的RFAConv
#     def __init__(self, in_channel, out_channel, kernel_size=3):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
#         self.get_weights = nn.Sequential(
#             nn.Conv2d(in_channel * (kernel_size ** 2), in_channel * (kernel_size ** 2), kernel_size=1,
#                       groups=in_channel),
#             nn.BatchNorm2d(in_channel * (kernel_size ** 2)))
#
#         self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=kernel_size)
#         self.bn = nn.BatchNorm2d(out_channel)
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         unfold_feature = self.unfold(x)  # 获得感受野空间特征  b c*kernel**2,h*w
#         x = unfold_feature
#         data = unfold_feature.unsqueeze(-1)
#         weight = self.get_weights(data).view(b, c, self.kernel_size ** 2, h, w).permute(0, 1, 3, 4, 2).softmax(-1)
#         weight_out = rearrange(weight, 'b c h w (n1 n2) -> b c (h n1) (w n2)', n1=self.kernel_size,
#                                n2=self.kernel_size)  # b c h w k**2 -> b c h*k w*k
#         receptive_field_data = rearrange(x, 'b (c n1) l -> b c n1 l', n1=self.kernel_size ** 2).permute(0, 1, 3,
#                                                                                                         2).reshape(b, c,
#                                                                                                                    h, w,
#                                                                                                                    self.kernel_size ** 2)  # b c*kernel**2,h*w ->  b c h w k**2
#         data_out = rearrange(receptive_field_data, 'b c h w (n1 n2) -> b c (h n1) (w n2)', n1=self.kernel_size,
#                              n2=self.kernel_size)  # b c h w k**2 -> b c h*k w*k
#         conv_data = data_out * weight_out
#         conv_out = self.conv(conv_data)
#         return self.act(self.bn(conv_out))


# 在Visdrone数据集中，基于Unfold和Group Conv的RFAConv进行了对比实验，检测模型为YOLOv5n，学习率为0.1，batch-size为8，epoch为300，其他超参数为默认参数。 其中RFAConv替换了C3瓶颈中的3*3卷积运算。
# 实验表明，基于Group Conv的RFAConv性能更好，因为Unfold提取感受野空间特征时，一定程度上消耗时间比较严重。因此全文选择了Group Conv的方法进行实验，并通过这种方式对CBAM和CA进行改进。
""" 一个小白写给读者的一些启发： 
（1）基于局部窗口的自注意力，最后通过softmax进行加权，然后进行sum融合特征。以这种角度理解RFAConv，同样通过Softmax进行加权，然后通过卷积核参数进行sum融合局部窗口的信息。
那么是否可以将局部窗口的自注意力最后的sum也通过高效的卷积参数或者全连接参数进行融合。
（2）除去论文外的其他的空间注意力是否可以把关注度放到感受野空间特征中呢，我觉得这是可行的。
"""


class SE(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[0:2]
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class RFCBAMConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        if kernel_size % 2 == 0:
            assert ("the kernel_size must be  odd.")
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
            )
        self.get_weight = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())
        self.se = SE(in_channel)

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel), nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        channel_attention = self.se(x)
        generate_feature = self.generate(x)

        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        unfold_feature = generate_feature * channel_attention
        max_feature, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feature = torch.mean(generate_feature, dim=1, keepdim=True)
        receptive_field_attention = self.get_weight(torch.cat((max_feature, mean_feature), dim=1))
        conv_data = unfold_feature * receptive_field_attention
        return self.conv(conv_data)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class RFCAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
        super(RFCAConv, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp, inp * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                                                stride=stride, groups=inp,
                                                bias=False),
                                      nn.BatchNorm2d(inp * (kernel_size ** 2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride=kernel_size))

    def forward(self, x):
        # logger.info(f"\nRFCAConv ==> 输入张量形状: {x.shape}")
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        h, w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = self.conv(generate_feature * a_w * a_h)
        #logger.info(f"RFCAConv ==> 输出张量形状: {out.shape}")
        return out

def test_default():
    # 示例输入张量
    input_tensor = torch.randn(1, 64, 32, 32)  # 批大小为1，通道数为64，32x32的空间维度

    # RFCAConv
    rfca_conv = RFCAConv(inp=64, oup=128, kernel_size=3, stride=1)
    output_rfca_conv = rfca_conv(input_tensor)
    print(f"RFCAConv Input Shape: {input_tensor.shape}")
    print(f"RFCAConv Output Shape: {output_rfca_conv.shape}")

    # RFCBAMConv
    rfcbam_conv = RFCBAMConv(in_channel=64, out_channel=128, kernel_size=3, stride=1)
    output_rfcbam_conv = rfcbam_conv(input_tensor)
    print(f"RFCBAMConv Input Shape: {input_tensor.shape}")
    print(f"RFCBAMConv Output Shape: {output_rfcbam_conv.shape}")

    # RFAConv - Based on Group Convolution
    rfa_conv_group = RFAConv(in_channel=64, out_channel=128, kernel_size=3, stride=1)
    output_rfa_conv_group = rfa_conv_group(input_tensor)
    print(f"RFAConv (Group) Input Shape: {input_tensor.shape}")
    print(f"RFAConv (Group) Output Shape: {output_rfa_conv_group.shape}")

    # RFAConv - Based on Unfold
    rfa_conv_unfold = RFAConv(in_channel=64, out_channel=128, kernel_size=3)
    output_rfa_conv_unfold = rfa_conv_unfold(input_tensor)
    print(f"RFAConv (Unfold) Input Shape: {input_tensor.shape}")
    print(f"RFAConv (Unfold) Output Shape: {output_rfa_conv_unfold.shape}")


def test_custom():
    # 示例输入张量
    input_tensor = torch.randn(1, 16, 128, 128)  # 批大小为1，通道数为64，32x32的空间维度

    # RFCAConv
    # 1, 16, 128, 128
    rfca_conv = RFCAConv(inp=16, oup=16, kernel_size=3, stride=1)
    output_rfca_conv = rfca_conv(input_tensor)
    print(f"RFCAConv Input Shape: {input_tensor.shape}")
    print(f"RFCAConv Output Shape: {output_rfca_conv.shape}")

    # RFCBAMConv
    # rfcbam_conv = RFCBAMConv(in_channel=64, out_channel=128, kernel_size=3, stride=1)
    # output_rfcbam_conv = rfcbam_conv(input_tensor)
    # print(f"RFCBAMConv Input Shape: {input_tensor.shape}")
    # print(f"RFCBAMConv Output Shape: {output_rfcbam_conv.shape}")

    # RFAConv - Based on Group Convolution
    # rfa_conv_group = RFAConv(in_channel=64, out_channel=128, kernel_size=3, stride=1)
    # output_rfa_conv_group = rfa_conv_group(input_tensor)
    # print(f"RFAConv (Group) Input Shape: {input_tensor.shape}")
    # print(f"RFAConv (Group) Output Shape: {output_rfa_conv_group.shape}")

    # RFAConv - Based on Unfold
    # rfa_conv_unfold = RFAConv(in_channel=64, out_channel=128, kernel_size=3)
    # output_rfa_conv_unfold = rfa_conv_unfold(input_tensor)
    # print(f"RFAConv (Unfold) Input Shape: {input_tensor.shape}")
    # print(f"RFAConv (Unfold) Output Shape: {output_rfa_conv_unfold.shape}")

if __name__ == '__main__':
    # 示例输入张量
    input_tensor = torch.randn(1, 64, 32, 32)  # 批大小为1，通道数为64，32x32的空间维度
    input_tensor2 = torch.randn(1, 64, 32, 32)  # 批大小为1，通道数为64，32x32的空间维度

    # RFCAConv
    # 1, 16, 128, 128
    rfca_conv = RFCAConv(inp=64, oup=128, kernel_size=3, stride=1)
    output_rfca_conv = rfca_conv(input_tensor)
    print(f"RFCAConv Input Shape: {input_tensor.shape}")
    print(f"RFCAConv Output Shape: {output_rfca_conv.shape}")

    # RFCBAMConv
    rfcbam_conv = RFCBAMConv(in_channel=64, out_channel=128, kernel_size=3, stride=1)
    output_rfcbam_conv = rfcbam_conv(input_tensor)
    print(f"RFCBAMConv Input Shape: {input_tensor.shape}")
    print(f"RFCBAMConv Output Shape: {output_rfcbam_conv.shape}")

    # RFAConv - Based on Group Convolution
    rfa_conv_group = RFAConv(in_channel=64, out_channel=128, kernel_size=3, stride=1)
    output_rfa_conv_group = rfa_conv_group(input_tensor)
    print(f"RFAConv (Group) Input Shape: {input_tensor.shape}")
    print(f"RFAConv (Group) Output Shape: {output_rfa_conv_group.shape}")

    # RFAConv - Based on Unfold
    rfa_conv_unfold = RFAConv(in_channel=64, out_channel=128, kernel_size=3)
    output_rfa_conv_unfold = rfa_conv_unfold(input_tensor)
    print(f"RFAConv (Unfold) Input Shape: {input_tensor.shape}")
    print(f"RFAConv (Unfold) Output Shape: {output_rfa_conv_unfold.shape}")
