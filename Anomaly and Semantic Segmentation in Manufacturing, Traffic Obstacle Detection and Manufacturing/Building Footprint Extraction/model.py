import torch 
import torch.nn as nn


def double_conv(in_c, out_c):
    
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
        self.dconv1= double_conv(in_channels,64)
        self.dconv2= double_conv(64,128)
        self.dconv3= double_conv(128,256)
        self.dconv4= double_conv(256,512)
        self.dconv5= double_conv(512,1024)
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
        print(dec_x_9.size())
        return dec_x_9