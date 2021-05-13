# U-Net: Convolutional Networks for Biomedical Image Segmentation 
# paper link: https://arxiv.org/abs/1505.04597

import torch
import torch.nn as nn

pretrained_model_path = {'pascal': 'weights/pretrained_model.pth'}

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm', deconv=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nker = nker
        self.norm = norm
        self.deconv = deconv

        # down-sampling
        self.enc1 = UNetEnc(self.in_channels, self.nker, self.norm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2 = UNetEnc(self.nker, self.nker * 2, self.norm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3 = UNetEnc(self.nker * 2, self.nker * 4, self.norm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4 = UNetEnc(self.nker * 4, self.nker * 8, self.norm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # center
        self.center = UNetEnc(self.nker * 8, self.nker * 8, self.norm)

        # up-sampling
        self.dec4 = UNetDec(self.nker * 8 * 2, self.nker * 4, self.deconv)
        self.dec3 = UNetDec(self.nker * 4 * 2, self.nker * 2, self.deconv)
        self.dec2 = UNetDec(self.nker * 2 * 2, self.nker * 1, self.deconv)
        self.dec1 = UNetDec(self.nker * 1 * 2, self.nker * 1, self.deconv)
        
        # final layer
        self.fc =  nn.Conv2d(in_channels=self.nker, out_channels=self.out_channels,
                            kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1 = self.enc1(x)
        maxpool1 = self.maxpool1(enc1)

        enc2 = self.enc2(maxpool1)
        maxpool2 = self.maxpool2(enc2)

        enc3 = self.enc3(maxpool2)
        maxpool3 = self.maxpool3(enc3)

        enc4 = self.enc4(maxpool3)
        maxpool4 = self.maxpool4(enc4)

        center = self.center(maxpool4)

        dec4 = self.dec4(enc4, center)
        dec3 = self.dec3(enc3, dec4)
        dec2 = self.dec2(enc2, dec3)
        dec1 = self.dec1(enc1, dec2)

        y = self.fc(dec1)

        return y        
        
## UNet Encoder for down-sampling: CBR x 2
class UNetEnc(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bnorm'):
        super(UNetEnc, self).__init__()

        layers = []

        # CBR(1): Convolution - Batch Normalization - ReLU
        layers += [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]
        if norm is not None:
            layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.ReLU()]

        # CBR(2): Convolution - Batch Normalization - ReLU
        layers += [nn.Conv2d(out_channels, out_channels, 3, 1, 1)]
        if norm is not None:
            layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.ReLU()]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

## UNet Decoder for up-sampling
class UNetDec(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNetDec, self).__init__()
        if deconv:
            self.dec1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=True)
        else:
            self.dec1 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x1, x2):
        x1 = self.dec1(x1)
        return self.dec2(torch.cat((x1, x2), dim=1))
        

def unet(in_channels=3, num_classes=21, nker=64, norm='bnorm', deconv=True, pretrained=False):
    if pretrained:
        model_path = pretrained_model_path['pascal']
        model = UNet(in_channels=in_channels, out_channels=num_classes, nker=nker, norm=norm, deconv=deconv)
        checkpoint = torch.load(model_path)
        weights = checkpoint['state_dict']
        weights['notinuse'] = weights.pop('final.weight')
        weights['notinuse2'] = weights.pop('final.bias')
        model.load_state_dict(weights, strict=False)
    else:
        model = UNet(in_channels=in_channels, out_channels=num_classes, nker=nker, norm=norm, deconv=deconv)

    return model