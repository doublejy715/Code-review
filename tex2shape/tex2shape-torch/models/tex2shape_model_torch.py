import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import hickle as hkl

from models.base_model import BaseModel

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("check cuda available!!")

class Tex2ShapeModel(BaseModel):
    def __init__(self, input_shape=(512, 512, 3), output_dims=6,
                 kernel_size=3, dropout_rate=0, bn=True, final_layer=None):
        super(Tex2ShapeModel,self).__init__()
        self.input_shape = input_shape
        self.output_dims = input_shape[2] if output_dims is None else output_dims
        self.kernel_size = (kernel_size, kernel_size)
        self.dropout_rate = dropout_rate
        self.bn = bn
        self.final_layer = final_layer
        
        self.parameter = hkl.load('../weights/tex2shape_weights.hdf5')

        # self.build_model()
        # layer value
        self.filters = 64
        self.f_size = 4

        # define layers
        self.dConv1 = nn.Conv2d(3,self.filters*1, self.f_size, stride=2, padding=1).to(device)
        self.dConv2 = nn.Conv2d(self.filters*1,self.filters*2,self.f_size, stride=2, padding=1).to(device)
        self.dConv3 = nn.Conv2d(self.filters*2,self.filters*4,self.f_size, stride=2, padding=1).to(device)
        self.dConv4 = nn.Conv2d(self.filters*4,self.filters*8,self.f_size, stride=2, padding=1).to(device)
        self.dConv5 = nn.Conv2d(self.filters*8,self.filters*8,self.f_size, stride=2, padding=1).to(device)
        self.dConv6 = nn.Conv2d(self.filters*8,self.filters*8,self.f_size, stride=2, padding=1).to(device)
        self.dConv7 = nn.Conv2d(self.filters*8,self.filters*8,self.f_size, stride=2, padding=1).to(device)

        self.UPsample = nn.Upsample(scale_factor=2).to(device)
        self.uConv1 = nn.Conv2d(self.filters*8,self.filters*8,self.f_size, stride=1, padding='same').to(device)
        self.uConv2 = nn.Conv2d(self.filters*8*2,self.filters*8,self.f_size, stride=1, padding='same').to(device)
        self.uConv3 = nn.Conv2d(self.filters*8*2,self.filters*8,self.f_size, stride=1, padding='same').to(device)
        self.uConv4 = nn.Conv2d(self.filters*8*2,self.filters*4,self.f_size, stride=1, padding='same').to(device)
        self.uConv5 = nn.Conv2d(self.filters*4*2,self.filters*2,self.f_size, stride=1, padding='same').to(device)
        self.uConv6 = nn.Conv2d(self.filters*2*2,self.filters*1,self.f_size, stride=1, padding='same').to(device)

        self.LeakyReLU = nn.LeakyReLU(0.2).to(device)
        self.ReLU = nn.ReLU().to(device)
        self.DropOut = nn.Dropout(self.dropout_rate).to(device)

        self.lastConv = nn.Conv2d(self.filters*2,self.output_dims,self.f_size,padding='same').to(device)

    def forward(self, x):
        x = x.reshape(self.input_shape)
        trans = transforms.ToTensor()
        x = trans(x).unsqueeze(0).to(device, dtype=torch.float32)

        x = self._unet_core(x)
        # bias option?
        x = self.lastConv(x)

        # self.final_layer == None 
        # if self.final_layer:
        #     x = self.final_layer(x)

        return x

    def _unet_core(self, d0):

        # down sampling conv2d
        def down_layer(input_layer, bn=True):
            d = self.LeakyReLU(input_layer)
            if bn:
                dBN = nn.BatchNorm2d(d.shape[1],momentum=0.8).to(device)
                d = dBN(d)
            return d

        # up sampling conv2d
        def up_layer(layer_input, skip_input):
            u = self.ReLU(layer_input)
            if self.dropout_rate:
                u = self.DropOut(u,p=self.dropout_rate)
            if self.bn:
                uBN = nn.BatchNorm2d(u.shape[1],momentum=0.8).to(device)
                u = uBN(u)
            u = torch.cat([u,skip_input],dim=1)
            return u

        # Stack Downsampling layer
        d1 = down_layer(self.dConv1(d0))
        d2 = down_layer(self.dConv2(d1))
        d3 = down_layer(self.dConv3(d2))
        d4 = down_layer(self.dConv4(d3)) 
        d5 = down_layer(self.dConv5(d4)) 
        d6 = down_layer(self.dConv6(d5)) 
        d7 = down_layer(self.dConv7(d6))

        # Stack Upsampling layer
        u1 = up_layer(self.uConv1(self.UPsample(d7)),d6)
        u2 = up_layer(self.uConv2(self.UPsample(u1)),d5)
        u3 = up_layer(self.uConv3(self.UPsample(u2)),d4)
        u4 = up_layer(self.uConv4(self.UPsample(u3)),d3)
        u5 = up_layer(self.uConv5(self.UPsample(u4)),d2)
        u6 = up_layer(self.uConv6(self.UPsample(u5)),d1)

        u7 = self.UPsample(u6)


        return u7


if __name__ == "__main__":
    model = Tex2ShapeModel()
    model.summary()