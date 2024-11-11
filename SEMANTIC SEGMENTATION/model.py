import torch.nn as nn
import torch
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        #encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        #bottleneck
        self.bottleneck = self.conv_block(256, 512)

        #decoder
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        self.out = nn.Conv2d(64, 1, kernel_size = 1, padding = 0)


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU()
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))

        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

        dec3 = self.dec3(torch.cat([F.interpolate(bottleneck, scale_factor = 2), enc3], dim = 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, scale_factor = 2), enc2], dim = 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, scale_factor = 2), enc1], dim = 1))


        out = self.out(dec1)
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)  # Resize output to match input size
    
        return torch.sigmoid(out)
    



        