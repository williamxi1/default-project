import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module import SNConv2d, SNLinear, GBlock, DBlock, DBlockOptimized

import torch.nn.utils.spectral_norm as SpectralNorm


class Generator32(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, ngf=256, bottom_width=4):
        super().__init__()

        self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf, upsample=True)
        self.block3 = GBlock(ngf, ngf, upsample=True)
        self.block4 = GBlock(ngf, ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(ngf)
        self.c5 = nn.Conv2d(ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = self.c5(h)
        y = torch.tanh(h)
        return y

class ConditionalGenerator32(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, nc=10, ngf=256, bottom_width=4):
        super().__init__()
        self.embed = nn.Embedding(nc, nz)
        self.l1 = nn.Linear(2*nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf, upsample=True)
        self.block3 = GBlock(ngf, ngf, upsample=True)
        self.block4 = GBlock(ngf, ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(ngf)
        self.c5 = nn.Conv2d(ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x, c):
        c_emb = self.embed(c)
        x = torch.cat((x, c_emb), dim=1)
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = self.c5(h)
        y = torch.tanh(h)
        return y

class Discriminator32(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=128):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf)
        self.block2 = DBlock(ndf, ndf, downsample=True)
        self.block3 = DBlock(ndf, ndf, downsample=False)
        self.block4 = DBlock(ndf, ndf, downsample=False)
        self.l5 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l5(h)
        return y

class ConditionalDiscriminator32(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=128, nc=10):
        super().__init__()
        self.embed = nn.Embedding(nc, 32*32)
        self.block1 = DBlockOptimized(3 +1, ndf)
        self.block2 = DBlock(ndf, ndf, downsample=True)
        self.block3 = DBlock(ndf, ndf, downsample=False)
        self.block4 = DBlock(ndf, ndf, downsample=False)
        self.l5 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x, c):
        c_emb = self.embed(c)
        c_emb = c_emb.reshape((c.shape[0],1,32,32))
        h = torch.cat((x, c_emb), dim = 1)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l5(h)
        return y

class Generator64(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, ngf=1024, bottom_width=4):
        super().__init__()

        self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf >> 1, upsample=True)
        self.block3 = GBlock(ngf >> 1, ngf >> 2, upsample=True)
        self.block4 = GBlock(ngf >> 2, ngf >> 3, upsample=True)
        self.block5 = GBlock(ngf >> 3, ngf >> 4, upsample=True)
        self.b6 = nn.BatchNorm2d(ngf >> 4)
        self.c6 = nn.Conv2d(ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = self.c6(h)
        y = torch.tanh(h)
        return y

class ConditionalGenerator64(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, nc = 120, ngf=1024, bottom_width=4, att=False):
        super().__init__()

        self.att = att
        self.embed = nn.Embedding(nc, nz)
        self.l1 = nn.Linear(2*nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf >> 1, upsample=True)
        self.block3 = GBlock(ngf >> 1, ngf >> 2, upsample=True)
        if att:
            self.attention = Attention(ngf >> 2)
        self.block4 = GBlock(ngf >> 2, ngf >> 3, upsample=True)
        self.block5 = GBlock(ngf >> 3, ngf >> 4, upsample=True)
        self.b6 = nn.BatchNorm2d(ngf >> 4)
        self.c6 = nn.Conv2d(ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)

    def forward(self, x, c):
        c_emb = self.embed(c)
        x = torch.cat((x, c_emb), dim=1)
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        if self.att:
            h = self.attention(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = self.c6(h)
        y = torch.tanh(h)
        return y

class Discriminator64(nn.Module):

    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=1024):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf >> 4)
        self.block2 = DBlock(ndf >> 4, ndf >> 3, downsample=True)
        self.block3 = DBlock(ndf >> 3, ndf >> 2, downsample=True)
        self.block4 = DBlock(ndf >> 2, ndf >> 1, downsample=True)
        self.block5 = DBlock(ndf >> 1, ndf, downsample=True)
        self.l6 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l6(h)
        return y

class ConditionalDiscriminator64(nn.Module):

    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=1024, nc=120, att=False):
        super().__init__()

        self.att = att
        self.embed = nn.Embedding(nc, 64*64)
        self.block1 = DBlockOptimized(3 + 1, ndf >> 4)
        if att:
            self.attention = Attention(ndf >> 4)
        self.block2 = DBlock(ndf >> 4, ndf >> 3, downsample=True)
        self.block3 = DBlock(ndf >> 3, ndf >> 2, downsample=True)
        self.block4 = DBlock(ndf >> 2, ndf >> 1, downsample=True)
        self.block5 = DBlock(ndf >> 1, ndf, downsample=True)
        self.l6 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def forward(self, x, c):
        c_emb = self.embed(c)
        c_emb = c_emb.reshape((c.shape[0],1,64,64))
        h = torch.cat((x, c_emb), dim = 1)
        h = self.block1(h)
        if self.att:
            h = self.attention(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l6(h)
        return y







def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
            
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta    = nn.utils.spectral_norm(
                        nn.Conv2d(channels, channels//8, kernel_size=1, padding=0, bias=True)).apply(init_weight)
        self.phi      = nn.utils.spectral_norm(
                        nn.Conv2d(channels, channels//8, kernel_size=1, padding=0, bias=True)).apply(init_weight)
        self.g        = nn.utils.spectral_norm(
                        nn.Conv2d(channels, channels//2, kernel_size=1, padding=0, bias=True)).apply(init_weight)
        self.o        = nn.utils.spectral_norm(
                        nn.Conv2d(channels//2, channels, kernel_size=1, padding=0, bias=True)).apply(init_weight)
        self.gamma    = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def forward(self, inputs):
        batch,c,h,w = inputs.size()
        theta = self.theta(inputs) #->(*,c/8,h,w)
        phi   = F.max_pool2d(self.phi(inputs), [2,2]) #->(*,c/8,h/2,w/2)
        g     = F.max_pool2d(self.g(inputs), [2,2]) #->(*,c/2,h/2,w/2)
        
        theta = theta.view(batch, self.channels//8, -1) #->(*,c/8,h*w)
        phi   = phi.view(batch, self.channels//8, -1) #->(*,c/8,h*w/4)
        g     = g.view(batch, self.channels//2, -1) #->(*,c/2,h*w/4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1) #->(*,h*w,h*w/4)
        o    = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch,self.channels//2,h,w)) #->(*,c,h,w)
        return self.gamma*o + inputs
class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channel, affine=False)  # no learning parameters
        self.embed = nn.Linear(n_condition, in_channel * 2) # part of w and part of bias

        nn.init.orthogonal_(self.embed.weight.data[:, :in_channel], gain=1)
        self.embed.weight.data[:, in_channel:].zero_()

    def forward(self, inputs, label):
        out = self.bn(inputs)
        embed = self.embed(label.float())
        gamma, beta = embed.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = (1+gamma) * out + beta
        return out

class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels,hidden_channels=None, upsample=False,n_classes = 0, leak = 0):
        super(ResBlockGenerator, self).__init__()
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.upsample = upsample
        self.learnable_sc = (in_channels != out_channels) or upsample
        
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1,bias = True)).apply(init_weight)
        self.conv2 = SpectralNorm(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1,bias = True)).apply(init_weight)
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,bias = True)).apply(init_weight)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.n_cl = n_classes
        if n_classes == 0:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(hidden_channels)
        else:
            self.bn1 = ConditionalNorm(in_channels,n_classes)
            self.bn2 = ConditionalNorm(hidden_channels,n_classes)

        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 
    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = self.upsampling(x)
                x = self.conv3(x)
            else:
                x = self.conv3(x)
            return x
        else:
            return x

    def forward(self, x,y=None):
        if y is not None:
            out = self.activation(self.bn1(x,y))
        else:
            out = self.activation(self.bn1(x))

        if self.upsample:
             out = self.upsampling(out)

        out = self.conv1(out)
        if y is not None:
            out = self.activation(self.bn2(out,y))
        else:
            out = self.activation(self.bn2(out))

        out = self.conv2(out)
        out_res = self.shortcut(x)
        return out + out_res


class ConditionalBNGenerator64(nn.Module):
    def __init__(self,z_dim =128,channels=3,ch = 64,n_classes = 120,leak = 0,att = False):
        super(ConditionalBNGenerator64, self).__init__()

        self.ch = ch
        self.n_classes = n_classes
        self.att = att
        self.dense = SpectralNorm(nn.Linear(z_dim, 4 * 4 * ch*8)).apply(init_weight)
        self.final = SpectralNorm(nn.Conv2d(ch, channels, kernel_size=3, padding=1,bias = True)).apply(init_weight)

        self.block1 = ResBlockGenerator(ch*8, ch*8,upsample=True,n_classes = n_classes,leak = leak)
        self.block2 = ResBlockGenerator(ch*8, ch*4,upsample=True,n_classes = n_classes,leak = leak)
        self.block3 = ResBlockGenerator(ch*4, ch*2,upsample=True,n_classes = n_classes,leak = leak)
        if att:
            self.attention = Attention(ch*2)
        self.block4 = ResBlockGenerator(ch*2, ch,upsample=True,n_classes = n_classes,leak = leak)

        self.bn = nn.BatchNorm2d(ch)
        if leak > 0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()

    def forward(self, z,y=None):
        h = self.dense(z).view(-1,self.ch*8, 4, 4)
        h = self.block1(h,y)
        h = self.block2(h, y)
        h = self.block3(h, y)
        if self.att:
            h = self.attention(h)
        h = self.block4(h,y)
        h = self.bn(h)
        h = self.activation(h)
        h = self.final(h)
        return nn.Tanh()(h)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False,hidden_channels=None,leak = 0):
        super(ResBlockDiscriminator, self).__init__()
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1,bias = True)).apply(init_weight)
        self.conv2 = SpectralNorm(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1,bias = True)).apply(init_weight)
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,bias = True)).apply(init_weight)

        self.learnable_sc = (in_channels != out_channels) or downsample
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 
        self.downsampling = nn.AvgPool2d(2)
        self.downsample = downsample

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.downsampling(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv3(x)
            if self.downsample:
                return self.downsampling(x)
            else:
                return x
        else:
            return x

    def forward (self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(nn.Module):

    def __init__(self, in_channels, out_channels,leak =0):
        super(OptimizedBlock, self).__init__()
        
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias = True)).apply(init_weight)
        self.conv2 = SpectralNorm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias = True)).apply(init_weight)
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,bias = True)).apply(init_weight)
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 
        
        self.model = nn.Sequential(
            self.conv1,
            self.activation,
            self.conv2,
            nn.AvgPool2d(2)  # stride = 2 ( default = kernel size)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.conv3
        )
    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ConditionalBNDiscriminator64(nn.Module):
    def __init__(self, channels=3,ch = 32,n_classes=120,leak =0,att = False):
        super(ConditionalBNDiscriminator64, self).__init__()
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()
            
        self.ch = ch
        self.att = att
        self.block1=OptimizedBlock(channels, ch,leak = leak)
        if att:
            self.attention = Attention(ch)
        self.block2=ResBlockDiscriminator(ch, ch*2, downsample=True,leak = leak)
        self.block3=ResBlockDiscriminator(ch*2 , ch*4,downsample=True,leak = leak)
        self.block4=ResBlockDiscriminator(ch*4, ch*8,downsample=True,leak = leak)
        self.block5=ResBlockDiscriminator(ch* 8, ch*16,leak = leak)
            
        self.fc =  SpectralNorm(nn.Linear(self.ch*16, 1)).apply(init_weight)
        if n_classes > 0:
            self.embed_y = nn.Embedding(n_classes,ch * 16).apply(init_weight)

    def forward(self, x,y=None):
        h = self.block1(x)
        if self.att:
            h = self.attention(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h,dim = (2,3))
        
        h = h.view(-1, self.ch*16)
        
        output = self.fc(h)
        if y is not None:
            output += torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)
        return output
