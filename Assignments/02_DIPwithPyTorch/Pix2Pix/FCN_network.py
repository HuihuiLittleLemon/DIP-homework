import torch.nn as nn


class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, 8, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        # FILL: add more CONV Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                8, 16, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                16, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                32, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                128, 256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(
                256, 512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # FILL: add ConvTranspose Layers
        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1
            ),  # Output channels: 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # Output channels: 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # Output channels: 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # Output channels: 8
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),  # Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(
                16, 8, kernel_size=4, stride=2, padding=1
            ),  # Output channels: 16
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(
                8, 3, kernel_size=4, stride=2, padding=1
            ),  # Output channels: 3 (for RGB)
            # Activation to keep output in range [0, 1] for images
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        # Decoder forward pass
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        output = self.deconv7(x)  # Output with RGB channels

        return output