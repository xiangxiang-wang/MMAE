import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from utils import patchify, unpatchify, forward_loss, compute_loss, image_padding, image_unpadding


class ChannelAttentionBlock(nn.Module):
    def __init__(self):
        super(ChannelAttentionBlock, self).__init__()
        self.ca1 = nn.AdaptiveAvgPool2d(1)
        self.ca2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x1, x2):
        EPSILON = 1e-10
        ca1 = self.ca1(x1)
        ca2 = self.ca1(x2)
        mask1 = torch.exp(ca1) / (torch.exp(ca2) + torch.exp(ca1) + EPSILON)
        mask2 = torch.exp(ca2) / (torch.exp(ca1) + torch.exp(ca2) + EPSILON)
        x1_a = mask1 * x1
        x2_a = mask2 * x2
        return x1_a, x2_a


class Patchembedding(nn.Module):
    def __init__(self, patch_size, in_chans=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = patchify(x, patch_size=self.patch_size)
        # x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class MaskedAutoencoder(nn.Module):
    def __init__(self, patch_size=16, in_chans=3,
                 embed_dim=256, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(MaskedAutoencoder, self).__init__()
        self.in_ch = in_chans
        self.patch_size = patch_size
        self.patch_dim = self.patch_size ** 2 * self.in_ch
        self.patch_attention = PatchAttention(patch_dim=self.patch_dim)
        self.patch_embed = Patchembedding(patch_size, in_chans, embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        self.sigmod = nn.Sigmoid()

    def random_masking(self, x, x2, pre_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # N, L, D = x.shape  # batch, length, dim

        mask = torch.mean(pre_mask, dim=2, keepdim=True)  ##N L 1
        # mask = torch.mean(pre_mask, dim=3, keepdim=True)  ##N L p2 1

        mask2 = torch.ones_like(mask)
        mask2 = mask2 - mask

        # ##红外和医学
        # mask2 = torch.ones_like(mask)

        x_masked = x * mask
        x2_masked = x2 * mask2
        # mask = torch.squeeze(mask, dim=-1)
        # mask2 = torch.squeeze(mask2, dim=-1)
        return x_masked, x2_masked

    def forward_encoder(self, x, x2, pre_mask):
        # # embed patches
        # x = self.embedding(x)
        # x2 = self.embedding(x2)
        # # add pos embed w/o cls token
        # x = x + self.pos_embed
        # x2 = x2 + self.pos_embed

        # embed patches
        x = self.patch_embed(x)  # N,L,D
        ## batch, length, dim
        ##length=num_patch,dim=dim_each_patch编码后的维度
        x2 = self.patch_embed(x2)

        # x = self.patch_attention(x)
        # x2 = self.patch_attention(x2)
        # add pos embed w/o cls token
        # x = x + self.pos_embed
        # x2 = x2 + self.pos_embed

        # x = x + pre_mask
        # pre_mask2 = torch.ones_like(pre_mask)
        # pre_mask2 = pre_mask2 - pre_mask
        # x2 = x2 + pre_mask2

        # masking: length -> length * mask_ratio
        x, x2 = self.random_masking(x, x2, pre_mask)

        for blk in self.blocks:
            x = blk(x)
            x2 = blk(x2)

        x = self.norm(x)
        x2 = self.norm(x2)

        return x, x2

    def forward_decoder(self, x, x2):

        # embed tokens
        x = self.decoder_embed(x)
        x2 = self.decoder_embed(x2)

        x = x + x2

        # add pos embed
        # x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward(self, imgs, imgs2, pre_mask):

        # imgs_patch = patchify(imgs, patch_size=self.patch_size)
        # imgs2_patch = patchify(imgs2, patch_size=self.patch_size)
        pre_mask = patchify(pre_mask, patch_size=self.patch_size)
        # imgs_att = self.patch_attention(imgs_patch)
        # imgs2_att = self.patch_attention(imgs2_patch)
        # imgs = unpatchify(imgs_att, patch_size=self.patch_size, in_ch=self.in_ch)
        # imgs2 = unpatchify(imgs2_att, patch_size=self.patch_size, in_ch=self.in_ch)
        _, _, h, w = imgs.shape
        h_ = h // self.patch_size
        w_ = w // self.patch_size

        x, x2 = self.forward_encoder(imgs, imgs2, pre_mask)
        pred = self.forward_decoder(x, x2)  # [N, L, p*p*3]
        print(pred.shape)
        pred = self.patch_attention(pred, h_, w_)
        pred = unpatchify(pred, patch_size=self.patch_size, in_ch=self.in_ch, img_h=h, img_w=w)
        # imgs = patchify(imgs, patch_size=self.patch_size, in_ch=self.in_ch)
        # imgs2 = patchify(imgs2, patch_size=self.patch_size, in_ch=self.in_ch)
        # loss = forward_loss(imgs_patch, imgs2_patch, pred, mask, mask2)
        # loss = 0
        return pred


class PatchAttention(nn.Module):
    def __init__(self, patch_dim):
        super(PatchAttention, self).__init__()

        self.patch_dim = patch_dim
        # self.in_ch = in_ch

        self.corr = nn.Conv2d(self.patch_dim, self.patch_dim, kernel_size=3, stride=1, padding=1, bias=False,
                              groups=self.patch_dim)

    def forward(self, x, h, w):
        N, L, patch_dim = x.shape
        # h = w = int(L ** .5)
        x = torch.transpose(x, 1, 2)  # N, patch_size**2 *3, L
        x = torch.reshape(x, (N, patch_dim, h, w))
        x = self.corr(x)
        x = self.corr(x)
        x = self.corr(x)
        x = torch.reshape(x, (N, patch_dim, L))
        x = torch.transpose(x, 1, 2)
        return x


class MMAE(nn.Module):
    def __init__(self, args):
        super(MMAE, self).__init__()
        in_ch = args.in_ch
        out_ch = args.out_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.patch_size = 16
        self.process1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU())

        self.process2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True),
                                      nn.ReLU())
        self.pool = nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=self.patch_size)
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=1, padding=2, bias=True),
                                   nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=7, stride=1, padding=3, bias=True),
                                   nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=9, stride=1, padding=4, bias=True),
                                   nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.conv5 = nn.Conv2d(2 * in_ch, out_ch, kernel_size=1, stride=1)

        self.mae = MaskedAutoencoder(in_chans=in_ch, patch_size=self.patch_size, embed_dim=12, depth=6, num_heads=4,
                                     decoder_embed_dim=24, decoder_depth=6, decoder_num_heads=4,
                                     mlp_ratio=4, norm_layer=nn.LayerNorm)
        # self.correct_attention = CorrectAttention(patch_size=self.patch_size, in_ch=in_ch, k=1)

        # self.channel_attention = ChannelAttention(in_ch)
        # self.channel_attention2 = ChannelAttention(out_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)

        self.reconstruction1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=9, stride=1, padding=4, bias=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True))

        self.reconstruction2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.reconstruction3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        self.reconstruction4 = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid())
        # self.ch_att = ChannelAttentionBlock()

    def forward(self, o1, o2):
        o1, m1, n1 = image_padding(o1, patch_size=self.patch_size)
        o2, m2, n2 = image_padding(o2, patch_size=self.patch_size)

        x1 = self.process1(o1)
        x2 = self.process1(o2)
        pre_mask = torch.cat([x1, x2], dim=1)
        pre_mask = self.process2(pre_mask)
        pre_mask = torch.mean(pre_mask, dim=1, keepdim=True)  ##b,1,h,w
        pre_mask = self.pool(pre_mask)  ##b,1,h//patch_size,w//patch_size
        s1 = torch.ones_like(pre_mask)
        s2 = torch.zeros_like(pre_mask)
        pre_mask = torch.where(pre_mask > 0.5, s1, s2)  ##b,1,h//patch_size,w//patch_size

        # pre_mask = self.up_sample(pre_mask)  ##b,1,h,w
        # pre_mask2 = torch.ones_like(pre_mask)  ##b,1,h,w
        # pre_mask2 = pre_mask2 - pre_mask  ##b,1,h,w

        x1 = o1
        x2 = o2

        x1 = self.conv1(x1)
        x11 = x1
        x2 = self.conv1(x2)
        x21 = x2
        x1_2_1 = torch.cat([x11, x21], dim=1)
        x1_2_1 = self.conv5(x1_2_1)

        x1 = self.conv2(x1)
        x12 = x1
        x2 = self.conv2(x2)
        x22 = x2
        x1_2_2 = torch.cat([x12, x22], dim=1)
        x1_2_2 = self.conv5(x1_2_2)

        x1 = self.conv3(x1)
        x13 = x1
        x2 = self.conv3(x2)
        x23 = x2
        x1_2_3 = torch.cat([x13, x23], dim=1)
        x1_2_3 = self.conv5(x1_2_3)

        x1 = self.conv4(x1)
        x14 = x1
        x2 = self.conv4(x2)
        x24 = x2
        x1_2_4 = torch.cat([x14, x24], dim=1)
        x1_2_4 = self.conv5(x1_2_4)

        output = self.mae(x1, x2, pre_mask)

        output = self.conv(output)
        # output = self.sigmoid(output)
        output = x1_2_4 + output
        output = self.reconstruction1(output)

        # output = x13 + x23 + output
        output = x1_2_3 + output
        output = self.reconstruction2(output)

        output = x1_2_2 + output
        output = self.reconstruction3(output)

        output = x1_2_1 + output
        output = self.reconstruction4(output)

        pre_mask = self.up_sample(pre_mask)  ##b,1,h,w
        pre_mask = image_unpadding(pre_mask, self.patch_size, m1, n1)
        output = image_unpadding(output, self.patch_size, m1, n1)
        return output, pre_mask
