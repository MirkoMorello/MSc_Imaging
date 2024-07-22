import torch
import torch.nn as nn
import torch.nn.functional as F

# this is the first version
class CustomNet(nn.Module):
    def __init__(self, num_classes=24):
        super(CustomNet, self).__init__()
        
        # encoder (downsampling)
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # bridge
        self.bridge = self.conv_block(512, 1024)
        
        # decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        # final layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        # encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.functional.max_pool2d(enc1, 2))
        enc3 = self.encoder3(nn.functional.max_pool2d(enc2, 2))
        enc4 = self.encoder4(nn.functional.max_pool2d(enc3, 2))
        
        # bridge
        bridge = self.bridge(nn.functional.max_pool2d(enc4, 2))
        
        # decoder
        dec4 = self.upconv4(bridge)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # final layer
        out = self.final_conv(dec1)
        
        return out


# this is the second version, the one we used

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.attention = AttentionBlock(out_channels)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.attention(x)
        return x, self.pool(x)

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()
        self.atrous_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.size()[2:]
        out1 = self.atrous_conv1(x)
        out2 = self.atrous_conv2(x)
        out3 = self.atrous_conv3(x)
        out4 = self.atrous_conv4(x)
        out5 = F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=True)
        out = torch.cat((out1, out2, out3, out4, out5), dim=1)
        out = self.conv1(out)
        out = self.bn1(out)
        return self.relu(out)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.attention = AttentionBlock(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.attention(x)
        return x

class CustomSegmentationModel(nn.Module):
    def __init__(self, num_classes=24):
        super(CustomSegmentationModel, self).__init__()
        
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)
        
        self.bridge = ASPPModule(256, 512)
        
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 32)
        
        self.final = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1, pool1 = self.enc1(x)
        enc2, pool2 = self.enc2(pool1)
        enc3, pool3 = self.enc3(pool2)
        enc4, pool4 = self.enc4(pool3)
        
        bridge = self.bridge(pool4)
        
        dec4 = self.dec4(bridge, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        
        return self.final(dec1)



# this was an experiment, it is not used in the paper    
    
# class AttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(AttentionBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         attention = self.sigmoid(self.conv(x))
#         return x * attention

# class EncoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(EncoderBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.attention = AttentionBlock(out_channels)

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.attention(x)
#         return x, self.pool(x)

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, skip_channels):
#         super(DecoderBlock, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
#         self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.attention = AttentionBlock(out_channels)

#     def forward(self, x, skip):
#         x = self.up(x)
#         x = torch.cat([x, skip], dim=1)
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.attention(x)
#         return x

# class SimplifiedImprovedModel(nn.Module):
#     def __init__(self, num_classes=24):
#         super(SimplifiedImprovedModel, self).__init__()
        
#         self.enc1 = EncoderBlock(3, 64)
#         self.enc2 = EncoderBlock(64, 128)
#         self.enc3 = EncoderBlock(128, 256)
#         self.enc4 = EncoderBlock(256, 512)
        
#         self.center = nn.Sequential(
#             nn.Conv2d(512, 1024, 3, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1024, 1024, 3, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             AttentionBlock(1024)
#         )
        
#         self.dec4 = DecoderBlock(1024, 512, 512)
#         self.dec3 = DecoderBlock(512, 256, 256)
#         self.dec2 = DecoderBlock(256, 128, 128)
#         self.dec1 = DecoderBlock(128, 64, 64)
        
#         self.final = nn.Conv2d(64, num_classes, kernel_size=1)

#     def forward(self, x):
#         enc1, pool1 = self.enc1(x)
#         enc2, pool2 = self.enc2(pool1)
#         enc3, pool3 = self.enc3(pool2)
#         enc4, pool4 = self.enc4(pool3)
        
#         center = self.center(pool4)
        
#         dec4 = self.dec4(center, enc4)
#         dec3 = self.dec3(dec4, enc3)
#         dec2 = self.dec2(dec3, enc2)
#         dec1 = self.dec1(dec2, enc1)
        
#         return self.final(dec1)