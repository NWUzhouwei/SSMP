from typing import no_type_check
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from .point import PointTransformer

cudnn.benchnark=True
from torch.nn import Conv1d

neg = 0.01
neg_2 = 0.2


def get_edge_features(x, k, num=-1, idx=None, return_idx=False):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]
    """
    B, dims, N = x.shape

    # batched pair-wise distance
    if idx is None:
        xt = x.permute(0, 2, 1)
        xi = -2 * torch.bmm(xt, x)
        xs = torch.sum(xt ** 2, dim=2, keepdim=True)
        xst = xs.permute(0, 2, 1)
        dist = xi + xs + xst  # [B, N, N]

        # get k NN id
        _, idx_o = torch.sort(dist, dim=2)
        idx = idx_o[:, :, 1:k + 1]  # [B, N, k]
        idx = idx.contiguous().view(B, N * k)

    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b])  # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors)  # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3)  # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k)  # [B, d, N, k]

    ee = torch.cat([central, neighbors - central], dim=1)
    assert ee.shape == (B, 2 * dims, N, k)

    if return_idx:
        return ee, idx
    return ee


class MyConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=torch.relu):
        super(MyConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        Conv = nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        # out = input
        out = gamma * out + beta

        return out

class EdgeBlock(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, attn=True):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin, Fout//2, 1),
            nn.BatchNorm2d(Fout//2),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv2d(Fout//2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_out = nn.Conv2d(Fout, Fout, [1, k], [1, 1])  # Fin, Fout, kernel_size, stride



    def forward(self, x):
        B, C, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        w = self.conv_w(x[:, C:, :, :])
        w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        x = self.conv_x(x)  # Bx2CxNxk
        x = x * w  # Bx2CxNxk

        x = self.conv_out(x)  # [B, 2*Fout, N, 1]

        x = x.squeeze(3)  # BxCxN

        return x

class SubspaceLayer(nn.Module):
    def __init__(self, dim, n_basis):
        super(SubspaceLayer, self).__init__()

        self.U = nn.Parameter(torch.empty(n_basis, dim))    # (6,96)
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(n_basis, 0, -1)]))    # (6)
        self.mu = nn.Parameter(torch.zeros(dim))    # (96)

    def forward(self, z):
        return (self.L * z) @ self.U + self.mu

class EigenBlock(nn.Module):
    def __init__(
        self,
        num_points,
        in_channels,
        n_basis
    ):
        super().__init__()

        self.convFeat = nn.Linear(256, n_basis, 1)
        self.projection = SubspaceLayer(dim=num_points*in_channels, n_basis=n_basis)
        self.subspace_conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, h, z):
        z = self.convFeat(z)
        phi = self.projection(z).view(h.shape)
        h = h + self.subspace_conv1(phi)
        return h

class self_attention(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.activation1 = torch.nn.GELU()
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, pos=None):
        src1 = self.input_proj(src1)
        b, c, _ = src1.shape
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src1 = self.norm13(src1)
        q=k=self.with_pos_embed(src1,pos)
        src12 = self.multihead_attn(query=q,
                                     key=k,
                                     value=src1)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)

        return src1

class SP_DecoderEigen3steps(nn.Module):
    def __init__(self, args):
        super(SP_DecoderEigen3steps, self).__init__()
        self.args = args
        self.nk = 64
        # self.nk = args.nk//2
        self.nz = 256
        # self.nz = args.nz
        Conv = nn.Conv1d

        dim = [3, 32, 64, 128]

        self.head1 = nn.Sequential(
            Conv(259, dim[-1], 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim[-1], dim[-1], 1),
            nn.LeakyReLU(neg, inplace=True),
        )


        self.head2 = nn.Sequential(
            Conv(288, dim[-1], 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim[-1], dim[-1], 1),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.head3 = nn.Sequential(
            Conv(320, dim[-1], 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim[-1], dim[-1], 1),
            nn.LeakyReLU(neg, inplace=True),
        )


        self.tail = nn.Sequential(
            self_attention(128, 64, dropout=0.0, nhead=8),
            # Conv1d(128, 64, 1),
            # nn.LeakyReLU(neg, inplace=True),
            Conv1d(64, 32, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(32, 3, 1),
            nn.Tanh()
        )

        self.point1 = PointTransformer(dim[0], dim[1])
        self.point2 = PointTransformer(dim[1], dim[2])
        self.point3 = PointTransformer(dim[2], dim[3])
        # self.EdgeConv1 = EdgeBlock(dim[0], dim[1], self.nk)
        self.adain1 = AdaptivePointNorm(dim[1], dim[-1])
        # self.EdgeConv2 = EdgeBlock(dim[1], dim[2], self.nk)
        self.adain2 = AdaptivePointNorm(dim[2], dim[-1])
        # self.EdgeConv3 = EdgeBlock(dim[2], dim[3], self.nk)
        self.adain3 = AdaptivePointNorm(dim[3], dim[-1])

        self.EigenBlock1 = EigenBlock(num_points=2048, in_channels=32, n_basis=18)
        self.EigenBlock2 = EigenBlock(num_points=2048, in_channels=64, n_basis=18)
        self.EigenBlock3 = EigenBlock(num_points=2048, in_channels=128, n_basis=18)


        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)
        self.lrelu3 = nn.LeakyReLU(neg_2)




    def forward(self, x, z):

        B,_,N =x.size()
        feat = z.view(-1, 1, 256).repeat([1, 2048, 1])
        feat = feat.permute(0, 2, 1)
        # feat=z
        # feat = z.unsqueeze(2).repeat(1, 1, 2048)
        # feat = z.unsqueeze(2).repeat(1, 1, self.args.number_points)
        style = torch.cat([x, feat], dim=1)
        #3+x
        style = self.head1(style)  # B,C,N
        x1=self.point1(x.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        # x1 = self.EdgeConv1(x)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)
        x1 = self.EigenBlock1(x1, z)


        style1 = torch.cat([x1, feat], dim=1)
        style1 = self.head2(style1)
        x2 = self.point2(x1.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style1)
        x2 = self.EigenBlock2(x2, z)


        style2= torch.cat([x2, feat], dim=1)
        style2 = self.head3(style2)
        x3 = self.point3(x2.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x3 = self.lrelu3(x3)
        x3 = self.adain3(x3, style2)
        x3 = self.EigenBlock3(x3, z)
        x1_o = self.tail(x3)

        x1_p = x1_o + x

        return x1_p

    def orthogonal_regularizer(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                UUT = layer.U @ layer.U.t()
                reg.append(
                    ((UUT - torch.eye(UUT.shape[0], device=UUT.device)) ** 2).mean()
                )
        return sum(reg) / len(reg)

if __name__=='__main__':
    input_tensor = torch.rand(24,128).cuda()
    point=torch.rand(24,3,2048).cuda()
    args=None
    model=SP_DecoderEigen3steps(args).cuda()
    x=model(point,input_tensor)
    print(x.shape)
