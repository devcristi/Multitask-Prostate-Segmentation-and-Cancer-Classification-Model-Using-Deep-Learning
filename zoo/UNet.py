import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_seg_classes=3, n_bin_classes=1):
        """
        n_channels: numărul de canale de intrare (pentru imagini grayscale folosiți 1)
        n_seg_classes: numărul de clase pentru segmentare
        n_bin_classes: numărul de clase pentru clasificarea binară (de obicei 1)
        """
        super(UNet, self).__init__()
        # Encoder:
        self.inc    = DoubleConv(n_channels, 64)           # x1: 64 canale
        self.down1  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))   # x2: 128 canale
        self.down2  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))  # x3: 256 canale
        self.down3  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))  # x4: 512 canale
        self.down4  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024)) # x5: 1024 canale
        self.down5  = nn.Sequential(nn.MaxPool2d(2), DoubleConv(1024, 2048))# x6 (bottleneck): 2048 canale

        # Ramura de clasificare binară:
        # Se folosește adaptive average pooling pe x6 (2048 canale)
        # Se adaugă un dropout de 0.5 înainte de stratul linear.
        self.bin_dropout = nn.Dropout(0.3)
        self.bin_outc = nn.Sequential(
            nn.Linear(2048, n_bin_classes)
        )

        # Decoder:
        # Up1: de la 2048 la 1024, concatenează cu x5 (1024) => 2048 canale, apoi DoubleConv
        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024 + 1024, 1024)
        # Up2: de la 1024 la 512, concatenează cu x4 (512) => 1024 canale, apoi DoubleConv
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512 + 512, 512)
        # Up3: de la 512 la 256, concatenează cu x3 (256) => 512 canale, apoi DoubleConv
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256 + 256, 256)
        # Up4: de la 256 la 128, concatenează cu x2 (128) => 256 canale, apoi DoubleConv
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128 + 128, 128)
        # Up5: de la 128 la 64, concatenează cu x1 (64) => 128 canale, apoi DoubleConv
        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(64 + 64, 64)

        # Stratul final pentru segmentare:
        self.seg_outc = nn.Conv2d(64, n_seg_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)         # [B, 64, H, W]
        x2 = self.down1(x1)      # [B, 128, H/2, W/2]
        x3 = self.down2(x2)      # [B, 256, H/4, W/4]
        x4 = self.down3(x3)      # [B, 512, H/8, W/8]
        x5 = self.down4(x4)      # [B, 1024, H/16, W/16]
        x6 = self.down5(x5)      # [B, 2048, H/32, W/32]

        # Clasificare binară:
        pooled = F.adaptive_avg_pool2d(x6, 1).view(x6.size(0), -1)  # [B, 2048]
        pooled = self.bin_dropout(pooled)
        bin_out = self.bin_outc(pooled)  # [B, n_bin_classes]

        # Decoder:
        x = self.up1(x6)                # [B, 1024, H/16, W/16]
        x = torch.cat([x, x5], dim=1)     # Concat: [B, 1024+1024=2048, H/16, W/16]
        x = self.conv1(x)               # [B, 1024, H/16, W/16]

        x = self.up2(x)                 # [B, 512, H/8, W/8]
        x = torch.cat([x, x4], dim=1)     # Concat: [B, 512+512=1024, H/8, W/8]
        x = self.conv2(x)               # [B, 512, H/8, W/8]

        x = self.up3(x)                 # [B, 256, H/4, W/4]
        x = torch.cat([x, x3], dim=1)     # Concat: [B, 256+256=512, H/4, W/4]
        x = self.conv3(x)               # [B, 256, H/4, W/4]

        x = self.up4(x)                 # [B, 128, H/2, W/2]
        x = torch.cat([x, x2], dim=1)     # Concat: [B, 128+128=256, H/2, W/2]
        x = self.conv4(x)               # [B, 128, H/2, W/2]

        x = self.up5(x)                 # [B, 64, H, W]
        x = torch.cat([x, x1], dim=1)     # Concat: [B, 64+64=128, H, W]
        x = self.conv5(x)               # [B, 64, H, W]

        seg_out = self.seg_outc(x)
        return seg_out, bin_out
