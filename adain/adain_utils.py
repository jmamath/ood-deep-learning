"""
Created on Tue Aug 17 07:55:22 2021

@author: JeanMichelAmath

inspired by https://github.com/naoto0804/pytorch-AdaIN
"""

import torch
from adain.adain_net import decoder, vgg
import torch.nn as nn
from torch.autograd import Variable

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

DECODER = decoder
VGG = vgg

DECODER.eval()
VGG.eval()

DECODER.load_state_dict(torch.load('adain/decoder.pth'))
VGG.load_state_dict(torch.load('adain/vgg_normalised.pth'))
VGG = nn.Sequential(*list(VGG.children())[:31])

VGG.to(device)
DECODER.to(device)

def augment(style_loader, batch_x, device, alpha=0.1):
    '''
    Here, we augment a bactch of data with a random batch of styles .
    They must have the same shape to broadcast the augmentation.
    Parameters
    ----------
    style_loader : Torch DataLoader
        set of synthetic images.
    batch_x : Torch Tensor 
        it contains the images to augment.
    alpha : Float
        Control the importance of the synthetic augmentation. The default is 0.1.

    Returns
    -------
    augmented : Torch Tensor
        it is batch of data augmented.

    '''
    envs,_ = next(iter(style_loader))
    #import pdb; pdb.set_trace()
    if batch_x.shape[0] < envs.shape[0]:
        # the last batch won't have the same dimension
        envs = envs[:batch_x.shape[0]]
    assert envs.shape == batch_x.shape, 'style and image batch should have the same shape'
    with torch.no_grad():
        augmented = style_transfer(VGG, DECODER, batch_x.to(device), envs.to(device),
                            alpha)
    return augmented  

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    """
    This function performs the style transfer

    Parameters
    ----------
    vgg : TYPE
        DESCRIPTION.
    decoder : TYPE
        DESCRIPTION.
    content : TYPE
        DESCRIPTION.
    style : TYPE
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.data.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.data.size()[:2] == style_feat.data.size()[:2])
    size = content_feat.data.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class AdainNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(AdainNet, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.data.size() == target.data.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.data.size() == target.data.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        t = adaptive_instance_normalization(self.encode(content), style_feats[-1])

        g_t = self.decoder(Variable(t.data, requires_grad=True))
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
