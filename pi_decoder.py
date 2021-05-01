import torch.nn as nn
import math
import torch

__all__ = ['effnetv2_s']

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            SiLU(),
        )

    def forward(self, x):
        return self.double_conv(x) 

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #SiLU(),
                nn.PReLU(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #SiLU(),
                nn.PReLU(hidden_dim),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #SiLU(),
                nn.PReLU(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                #SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        print('Using Model: PI-DECODER')

        self.cfgs = [
            # t, c, n, s, SE
            [1,  24,  1, 1, 0],  #24, 144, 160
            [4,  48,  2, 2, 0],  #, 48, 72, 80
            [4,  64,  2, 2, 0],   # 64, 36, 40
            [4, 128,  2, 2, 1],   #128, 18, 20  
            [6, 160,  2, 2, 1],   # 160, 18, 20   26 
           # [6, 272,  1, 2, 1],   # 272, 9, 10     41
        ]

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        layers.append(conv_1x1_bn(160, 320))

        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(640, 8) if width_mult > 1.0 else 1792
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        
        
        self.channels = [24,48,64,160]
        self.in_channels = [48, 96,128, 320]

        self.dconv4_2 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(320, 128,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(128),
            #SiLU(),
            nn.PReLU(128)
        )
        self.dconv3_2 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(128, 64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            #SiLU(),
            nn.PReLU(64)
        )
        self.dconv2_2 = nn.ConvTranspose2d(64, 48, 4, padding=1, stride=2)
        self.dconv2_1 = nn.ConvTranspose2d(64, 48, 4, padding=1, stride=2)
        self.dconv1_2 = nn.ConvTranspose2d(48, 24, 4, padding=1, stride=2)
        self.dconv1_1 = nn.ConvTranspose2d(48, 24, 4, padding=1, stride=2)
        self.dconv1_3 = nn.ConvTranspose2d(48, 24, 4, padding=1, stride=2)
        self.dconv0_2=nn.Sequential(
            nn.ConvTranspose2d(24,3,4,padding=1,stride=2),
            nn.BatchNorm2d(3),
            nn.PReLU(3)
        )  
        self.dconv0_1=nn.Sequential(
            nn.ConvTranspose2d(24,3,4,padding=1,stride=2),
            nn.BatchNorm2d(3),
            nn.PReLU(3)
        )  
        self.dconv0_3=nn.Sequential(
            nn.ConvTranspose2d(24,3,4,padding=1,stride=2),
            nn.BatchNorm2d(3),
            nn.PReLU(3)
        )
        
        self.invres4_2 = DoubleConv(256, 128)  
        self.invres3_2 = DoubleConv(128, 64)
        self.invres2_2 = DoubleConv(96, 48)
        self.invres2_1 = DoubleConv(96, 48)
        self.invres1_2 = DoubleConv(48, 24)
        self.invres1_1 = DoubleConv(48, 24)
        self.invres1_3 = DoubleConv(48, 24) 
        
        self.conv_score1 = nn.Conv2d(3, 1, 1)
        self.conv_score2 = nn.Conv2d(3, 1, 1)
        self.conv_score3 = nn.Conv2d(3, 1, 1)

        self._initialize_weights()

    def forward(self, x):
        for n in range(0, 2):
            x = self.features[n](x)
        d1 = x
        for n in range(2, 4):
            x = self.features[n](x)
        d2 = x
        for n in range(4, 6):
            x = self.features[n](x)
        d3 = x
        for n in range(6, 8):
            x = self.features[n](x)
        d4 = x
        for n in range(8,11):
            x = self.features[n](x)
        d5 = x

        up4_2 = self.invres4_2(torch.cat([d4, self.dconv4_2(d5)], dim=1))
        up3_2 = self.invres3_2(torch.cat([d3, self.dconv3_2(up4_2)], dim=1))
        up2_2 = self.invres2_2(torch.cat([d2, self.dconv2_2(up3_2)], dim=1))
        up2_1 = self.invres2_1(torch.cat([d2, self.dconv2_1(up3_2)], dim=1))
        up1_2 = self.invres1_2(torch.cat([d1, self.dconv1_2(up2_2)], dim=1))
        up1_1 = self.invres1_1(torch.cat([d1, self.dconv1_1(up2_1)], dim=1))
        up1_3 = self.invres1_3(torch.cat([d1, self.dconv1_3(up2_1)], dim=1))

        up0_1 = self.dconv0_1(up1_1)
        up0_2 = self.dconv0_2(up1_2)
        up0_3 = self.dconv0_3(up1_3)

        x1 = self.conv_score1(up0_1)
        x2 = self.conv_score2(up0_2)
        x3 = self.conv_score3(up0_3)

        return x1, x2, x3
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

def effnetv2_s(**kwargs):
    return EffNetV2(**kwargs)

if __name__ =="__main__":
    # from torchsummary import summary
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    # model = EffNetV2( ).to(device)
    # summary(model, input_size=(3, 480, 640))

    from torchstat import stat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    print(device)
    model = EffNetV2().to(device)
    stat(model, (3, 480, 640))