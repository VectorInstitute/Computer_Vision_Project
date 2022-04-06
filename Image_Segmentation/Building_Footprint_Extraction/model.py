import torch 
import torch.nn as nn

class double_conv2(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , out_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    # def forward(self, x1, x2):  # x1--up , x2 ---down
    #     x1 = self.up(x1)
    #     diffX = x1.size()[2] - x2.size()[2]
    #     diffY = x2.size()[3] - x1.size()[3]
    #     x1 = F.pad(x1, (
    #         diffY // 2, diffY - diffY // 2,
    #         diffX // 2, diffX - diffX // 2,))
    #     x = torch.cat([x2, x1], dim=1)
    #     x = self.conv(x)
    #     return x
    def forward(self, x1, x2):# x1--up , x2 ---down
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class up3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up3, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        # print(x1.shape)
        x1 = self.up(x1)
        # input is CHW
        diffY = x3.size()[2] - x1.size()[2]
        diffX = x3.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class up4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up4, self).__init__()
        if 0:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2 // 3, in_ch * 2 // 3, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm3d(in_ch * 2 // 3),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=2)
        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2, x3, x4):  # x1--up , x2 ---down
        # print(x1.shape)
        x1 = self.up(x1)
        # input is CHW
        diffY = x4.size()[2] - x1.size()[2]
        diffX = x4.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        #self.upsample = nn.Upsample(scale_factor=4)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        #x = self.upsample(x)
        x = self.conv(x)
        #x = F.sigmoid(x)
        return x


class double_conv(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_in(nn.Module):
    '''(conv-BN-ReLU)X2 :   in_ch  , in_ch , out_ch    '''

    def __init__(self, in_ch, out_ch):
        super(double_conv_in, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),  # True means cover the origin input
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv2(in_ch, out_ch) # double_conv_in

    def forward(self, x):
        x = self.conv(x)
        return x


cc = 32  # you can change it to 8, then the model can be more faster ,reaching 35 fps on cpu when testing


class UNETPlus(nn.Module):
    def __init__(self, n_channels, n_classes, mode='train'):
        super(UNETPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inconv = inconv(n_channels, cc)
        self.down1 = down(cc, 2 * cc)
        self.down2 = down(2 * cc, 4 * cc)
        self.down3 = down(4 * cc, 8 * cc)
        self.up1 = up(12 * cc, 4 * cc)
        self.up20 = up(6 * cc, 2 * cc)
        self.up2 = up3(8 * cc, 2 * cc)
        self.up30 = up(3 * cc, cc)
        self.up31 = up3(4 * cc, cc)
        self.up3 = up4(5 * cc, cc)
        self.outconv = outconv(cc, n_classes)
        self.mode = mode

    def forward(self, x):
        if self.mode == 'train':  # use the whole model when training
            x1 = self.inconv(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up1(x4, x3)
            x21 = self.up20(x3, x2)
            x = self.up2(x, x21, x2)
            x11 = self.up30(x2, x1)
            x12 = self.up31(x21, x11, x1)
            x = self.up3(x, x12, x11, x1)
            #output 0 1 2
            y2 = self.outconv(x)
            y0 = self.outconv(x11)
            y1 = self.outconv(x12)
            return (y0+ y1+y2)/3
        else:  # prune the model when testing
            x1 = self.inconv(x)
            x2 = self.down1(x1)
            x11 = self.up30(x2, x1)
            # output 0
            y0 = self.outconv(x11)
            return y0



def double_conv_unet(in_c, out_c):
    
    conv= nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size= 3, padding=(1, 1)),
        nn.ReLU(inplace= False),
        nn.Conv2d(out_c, out_c, kernel_size= 3, padding=(1, 1)),
        nn.ReLU(inplace= False)
    )
    return conv

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.maxpool_2x2 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.dconv1= double_conv_unet(in_channels,64)
        self.dconv2= double_conv_unet(64,128)
        self.dconv3= double_conv_unet(128,256)
        self.dconv4= double_conv_unet(256,512)
        self.dconv5= double_conv_unet(512,1024)
        self.num_of_classes = out_channels
        
        #Now, the first up convolution is performed followed by a double convolution to alter the number of channels of feature map.
        self.uptrans1= nn.ConvTranspose2d(
            in_channels= 1024,
            out_channels= 512,
            kernel_size= 2,
            stride= 2
        )
        
        self.upconv1= double_conv(1024,512)
        
        self.uptrans2= nn.ConvTranspose2d(
            in_channels= 512,
            out_channels= 256,
            kernel_size= 2,
            stride= 2
        )
        
        self.upconv2= double_conv(512, 256)
        
        self.uptrans3= nn.ConvTranspose2d(
            in_channels= 256,
            out_channels= 128,
            kernel_size= 2,
            stride= 2
        )
        
        self.upconv3= double_conv(256,128)
        
        self.uptrans4= nn.ConvTranspose2d(
            in_channels= 128,
            out_channels= 64,
            kernel_size= 2,
            stride= 2
        )
        
        self.upconv4= double_conv(128,64)
        
        self.out= nn.Conv2d(
            in_channels= 64,
            out_channels= self.num_of_classes,
            kernel_size= 1
        )
    
    def forward(self, image):
        
        #encoder
        enc_x_1= self.dconv1(image)
        enc_x_2= self.maxpool_2x2(enc_x_1)
        enc_x_3= self.dconv2(enc_x_2)
        enc_x_4= self.maxpool_2x2(enc_x_3)
        enc_x_5= self.dconv3(enc_x_4)
        enc_x_6= self.maxpool_2x2(enc_x_5)
        enc_x_7= self.dconv4(enc_x_6)
        enc_x_8= self.maxpool_2x2(enc_x_7)
        enc_x_9= self.dconv5(enc_x_8)
        
        #decoder
        dec_x_1= self.uptrans1(enc_x_9)
        dec_x_2 = self.upconv1(torch.cat([dec_x_1, enc_x_7],1))
        
        dec_x_3= self.uptrans2(dec_x_2)
        dec_x_4= self.upconv2(torch.cat([dec_x_3, enc_x_5],1))
        
        dec_x_5= self.uptrans3(dec_x_4)
        dec_x_6= self.upconv3(torch.cat([dec_x_5, enc_x_3],1))
        
        dec_x_7= self.uptrans4(dec_x_6)
        dec_x_8= self.upconv4(torch.cat([dec_x_7, enc_x_1],1))
        
        dec_x_9= self.out(dec_x_8)

        return dec_x_9