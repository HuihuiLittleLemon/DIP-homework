# FCN_network.py

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, conv_channel_base=64, img_channel=3):
        super(Generator, self).__init__()
        self.conv_channel_base = conv_channel_base

        # 下采样（编码器）
        self.e1 = nn.Conv2d(img_channel, conv_channel_base,
                            kernel_size=4, stride=2, padding=1)  # 不加激活和归一化
        self.e2 = self.down_block(
            conv_channel_base, conv_channel_base * 2)  # e2
        self.e3 = self.down_block(
            conv_channel_base * 2, conv_channel_base * 4)  # e3
        self.e4 = self.down_block(
            conv_channel_base * 4, conv_channel_base * 8)  # e4
        self.e5 = self.down_block(
            conv_channel_base * 8, conv_channel_base * 8)  # e5
        self.e6 = self.down_block(
            conv_channel_base * 8, conv_channel_base * 8)  # e6
        self.e7 = self.down_block(
            conv_channel_base * 8, conv_channel_base * 8)  # e7
        self.e8 = self.down_block(
            conv_channel_base * 8, conv_channel_base * 8, batch_norm=False)  # e8

        # 上采样（解码器）
        self.d1 = self.up_block(conv_channel_base * 8,
                                conv_channel_base * 8, dropout=True)  # d1
        self.d2 = self.up_block(conv_channel_base * 16,
                                conv_channel_base * 8, dropout=True)  # d2
        self.d3 = self.up_block(conv_channel_base * 16,
                                conv_channel_base * 8, dropout=True)  # d3
        self.d4 = self.up_block(conv_channel_base * 16,
                                conv_channel_base * 8)  # d4
        self.d5 = self.up_block(conv_channel_base * 16,
                                conv_channel_base * 4)  # d5
        self.d6 = self.up_block(conv_channel_base * 8,
                                conv_channel_base * 2)  # d6
        self.d7 = self.up_block(conv_channel_base * 4, conv_channel_base)  # d7
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(conv_channel_base * 2, img_channel,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def down_block(self, in_channels, out_channels, batch_norm=True):
        layers = [
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=2, padding=1)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def up_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 编码器
        e1 = self.e1(x)  # 注意：e1 不经过激活和批归一化
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        # 解码器 + 跳跃连接
        d1 = self.d1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.d7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        output = self.d8(d7)
        return output


class Discriminator(nn.Module):
    def __init__(self, conv_channel_base=64, img_channel=3):
        super(Discriminator, self).__init__()
        self.conv_channel_base = conv_channel_base

        self.layer1 = nn.Sequential(
            nn.Conv2d(img_channel * 2, conv_channel_base,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 不使用批归一化

        self.layer2 = self.disc_block(conv_channel_base, conv_channel_base * 2)
        self.layer3 = self.disc_block(
            conv_channel_base * 2, conv_channel_base * 4)
        self.layer4 = self.disc_block(
            conv_channel_base * 4, conv_channel_base * 8, stride=1)
        self.layer5 = nn.Conv2d(conv_channel_base * 8,
                                1, kernel_size=4, stride=1, padding=1)

    def disc_block(self, in_channels, out_channels, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, img, cond):
        x = torch.cat([img, cond], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x  # 输出未经过激活函数，在损失函数中使用 nn.BCEWithLogitsLoss()
