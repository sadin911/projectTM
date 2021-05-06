import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#          Encoder
##############################

class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding = 1)
        self.mxpool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(8,16,3,padding = 1)
        self.mxpool2 = nn.MaxPool2d((2,2))
        self.conv3 = nn.Conv2d(16,32,3,padding = 1)
        self.mxpool3 = nn.MaxPool2d((2,2))
        
    def forward(self, img):
        x = self.conv1(img)
        x = self.mxpool1(x)
        x = self.conv2(x)
        x = self.mxpool2(x)
        x = self.conv3(x)
        x = self.mxpool3(x)
        return x

Encoder = Encoder()
print(Encoder)
params = list(Encoder.parameters())
print(len(params))
print(params[0].size())

##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)