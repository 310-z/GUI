import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50
from .CA import CoordAtt
from .SE import SEAttention

class  Edge_thinning(nn.Sequential):
    def __init__(self,dim_in,dim_out):
        super(Edge_thinning, self).__init__()
        self.conv1  = nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True)
        self.Sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(kernel_size=1, ceil_mode=True)
        self.residual = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=0),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*2, dim_out, 1, 1, padding=0, bias=True),  # dim_out=256
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        [b, c, row, col] = x.size()
        x_1 = self.Sigmoid(x)
        x_1 = self.residual(x_1)
        x_1 = self.maxpool1(x_1)
        x_1 = F.interpolate(x_1, (row, col), None, 'bilinear', True)
        x = torch.cat([x_1, x], dim=1)
        x = self.conv_cat(x)
        x = F.interpolate(x, size=(row, col), mode='bilinear', align_corners=False)
        return x
#定义特征融合上采样模块
# -----------------------------------------#
class deepUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(deepUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, downsample_factor=16):
        super(DeepLab, self).__init__()
        self.resnet = resnet50()
        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]

        self.up_concat4 =deepUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = deepUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = deepUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = deepUp(in_filters[0], out_filters[0])
        self.aspp = ASPP(2048, 2048)
        self.se   = SEAttention(2048,2048)
        self.CA   = CoordAtt(2048,2048)
        self.etc  = Edge_thinning(2048,2048)

        self.cat_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(x)
        feat5 = self.aspp(feat5)
        x = self.etc(feat5)
        x_1 = self.CA(x)
        x_2 = self.se(x)
        x  = x_1 + x_2
        up4 = self.up_concat4(feat4, x)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        x  = self.cat_conv(up1)
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


