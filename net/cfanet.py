import torch
import torch.nn as nn
import torch.nn.functional as F
from net.ResNet import resnet50
from math import log
from net.Res2Net import res2net50_v1b_26w_4s



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
       
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        output = self.relu(x_cat + self.conv_res(x))
        
        return output

class feature_exctraction(nn.Module):
    def __init__(self, in_channel, depth, kernel):
        super(feature_exctraction, self).__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.kernel = kernel
        self.conv1 = nn.Sequential(nn.Conv2d(self.depth, self.depth, self.kernel, 1, (self.kernel - 1) // 2),
                                   nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, 1, 1, 0), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))

    def forward(self, x):

        conv1 = self.conv3(x)
        output = self.conv1(conv1)

        return output

    #def initialize(self):
     #   weight_init(self)

class SANet(nn.Module):

    def __init__(self, in_dim, coff):
        super(SANet, self).__init__()
        self.dim = in_dim
        self.coff = coff
        self.k = 9
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 4, (1, self.k), 1, (0, self.k // 2)), nn.BatchNorm2d(self.dim // 4), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 4, (self.k, 1), 1, (self.k // 2, 0)), nn.BatchNorm2d(self.dim // 4), nn.ReLU(inplace=True))
        self.conv2_1 = nn.Conv2d(self.dim // 4, 1, (self.k, 1), 1, (self.k // 2, 0))
        self.conv2_2 = nn.Conv2d(self.dim // 4, 1, (1, self.k), 1, (0, self.k // 2))
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(x)
        conv2_1 = self.conv2_1(conv1_1)
        conv2_2 = self.conv2_2(conv1_2)
        conv3 = torch.add(conv2_1, conv2_2)
        conv4 = torch.sigmoid(conv3)

        conv5 = conv4.repeat(1, self.dim // self.coff, 1, 1)

        return conv5

    #def initialize(self):
     #   weight_init(self)

class SENet(nn.Module):

    def __init__(self, in_dim, ratio=2):
        super(SENet, self).__init__()
        self.dim = in_dim
        self.fc = nn.Sequential(nn.Linear(in_dim, self.dim // ratio, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // ratio, in_dim, bias = False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y=F.adaptive_avg_pool2d(x, (1,1)).view(b,c)
        y= self.sigmoid(self.fc(y)).view(b,c,1,1)
 
        output = y.expand_as(x)

        return output
        

    #def initialize(self):
    #    weight_init(self)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2,
                              bias=False)  # infer a one-channel attention map

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True)  # [B, 1, H, W], average
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True)  # [B, 1, H, W], max
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1)  # [B, 2, H, W]
        att_map = torch.sigmoid(self.conv(ftr_cat))  # [B, 1, H, W]
        return att_map

    #def initialize(self):
    #    weight_init(self)

class Edge_detect(nn.Module):

    def __init__(self):
        super(Edge_detect, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        output = self.conv3(self.relu(conv2 + x))

        return output
        
class Fuse(nn.Module):

    def __init__(self):
        super(Fuse, self).__init__()
        self.se = SENet(64)        

    def forward(self, x0, x1, x2): # x1 and x2 are the adjacent features
        x1_up = F.upsample(x1, size=x0.size()[2:], mode='bilinear')
        x2_up = F.upsample(x2, size=x0.size()[2:], mode='bilinear')
        x_fuse = x0 + x1_up + x2_up
        x_w = self.se(x_fuse)
        output = x0 * x_w + x0
        
        return output

class CLFF(nn.Module):

    def __init__(self):
        super(CLFF, self).__init__()
        self.fuse1 = Fuse()
        self.fuse2 = Fuse()  
        self.fuse3 = Fuse()          
        self.conv1 = nn.Sequential(nn.Conv2d(64*3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        

    def forward(self, x0, x1, x2): # x1 and x2 are the adjacent features
    
        x_g3 = self.fuse1(x0, x1, x2)
        x_g4 = self.fuse1(x1, x0, x2)
        x_g4 = F.upsample(x_g4, size=x_g3.size()[2:], mode='bilinear')
        x_g5 = self.fuse1(x2, x0, x1)
        x_g5 = F.upsample(x_g5, size=x_g3.size()[2:], mode='bilinear')
        
        output = self.conv1(torch.cat([x_g3, x_g4, x_g5], dim=1))
        
        return output




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # if self.training:
        # self.initialize_weights()

        self.fem_layer5 = feature_exctraction(512, 64, 7)
        self.fem_layer4 = feature_exctraction(512, 64, 5)
        self.fem_layer3 = feature_exctraction(256, 64, 5)
        self.fem_layer2 = feature_exctraction(128, 64, 3)
        self.fem_layer1 = feature_exctraction(64, 64, 3)

        self.con1 = nn.Sequential(nn.Conv2d(64, 64, 1,1, 0),nn.BatchNorm2d(64),nn.ReLU(inplace = True))
        self.con2 = nn.Sequential(nn.Conv2d(256, 128, 1,1, 0),nn.BatchNorm2d(128),nn.ReLU(inplace = True))
        self.con3 = nn.Sequential(nn.Conv2d(512, 256, 1,1, 0),nn.BatchNorm2d(256),nn.ReLU(inplace = True))
        self.con4 = nn.Sequential(nn.Conv2d(1024, 512,1,1, 0),nn.BatchNorm2d(512),nn.ReLU(inplace = True))
        self.con5 = nn.Sequential(nn.Conv2d(2048, 512,1,1,  0),nn.BatchNorm2d(512),nn.ReLU(inplace = True))
        
        self.UE5 = GCM(64, 64)
        self.UE4 = GCM(64, 64)
        self.UE3_2 = GCM(64, 64)
        
        self.clff_d = CLFF()
        self.clff_s = CLFF()
        
        self.edge = Edge_detect()

        self.conv1 = nn.Conv2d(64, 1, 1, 1, 0)        
        self.conv2 = nn.Conv2d(64, 1, 1, 1, 0)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0),nn.BatchNorm2d(64),nn.ReLU(inplace = True))

    # model_state = torch.load('./models/resnet50-19c8e357.pth')
    # self.resnet.load_state_dict(model_state, strict=False)

    def forward(self, x):
        conv1, conv2, conv3, conv4, conv5  = self.resnet(x)
        
        x_size = x.size()

        conv1 = self.con1(conv1)
        conv2 = self.con2(conv2)
        conv3 = self.con3(conv3)
        conv4 = self.con4(conv4)
        conv5 = self.con5(conv5)        

        fem_layer5 = self.fem_layer5(conv5)
        fem_layer4 = self.fem_layer4(conv4)
        fem_layer3 = self.fem_layer3(conv3)
        fem_layer2 = self.fem_layer2(conv2)
        fem_layer1 = self.fem_layer1(conv1)


        # cross-attention
        fea5 = self.UE5(fem_layer5)
        fea4 = self.UE4(fem_layer4)
        fea3_2 = self.UE3_2(fem_layer3)
        
        fd = self.clff_d(fea3_2, fea4, fea5)
        s2 = self.conv1(fd)
        
        s2_map = s2.sigmoid()
        fea3_1 = F.upsample(s2_map, size=fem_layer3.size()[2:], mode='bilinear') * fem_layer3 + fem_layer3
        fea2 = F.upsample(s2_map, size=fem_layer2.size()[2:], mode='bilinear') * fem_layer2 + fem_layer2
        fea1 = F.upsample(s2_map, size=fem_layer1.size()[2:], mode='bilinear') * fem_layer1 + fem_layer1
        
        fs = self.clff_s(fea1, fea2, fea3_1)
        
        f = self.conv3(torch.cat([F.upsample(fd, size=fs.size()[2:], mode='bilinear'), fs], dim=1))
        edge = self.edge(f)
        s1 = self.conv2(f)       

        pre3 = F.upsample(edge, size=x.size()[2:], mode='bilinear')
        pre2 = F.upsample(s2, size=x.size()[2:], mode='bilinear')
        pre1 = F.upsample(s1, size=x.size()[2:], mode='bilinear')


        return pre1, pre2, pre3.sigmoid()
