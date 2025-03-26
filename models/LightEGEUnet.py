import torch
import torch.nn as nn
import torch.nn.functional as F

# LayerNorm3D module
class LayerNorm3d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight[:, None, None, None] + self.bias[:, None, None, None]
        return x

# GroupAggregationBridge3D module
class GroupAggregationBridge3D(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.dim_xl = dim_xl
        self.pre_project = nn.Conv3d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm3d(group_size + 1),
            nn.Conv3d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size-1)*(d_list[0]-1))//2,
                      dilation=d_list[0], groups=group_size + 1)
        )
        self.g1 = nn.Sequential(
            LayerNorm3d(group_size + 1),
            nn.Conv3d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size-1)*(d_list[1]-1))//2,
                      dilation=d_list[1], groups=group_size + 1)
        )
        self.g2 = nn.Sequential(
            LayerNorm3d(group_size + 1),
            nn.Conv3d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size-1)*(d_list[2]-1))//2,
                      dilation=d_list[2], groups=group_size + 1)
        )
        self.g3 = nn.Sequential(
            LayerNorm3d(group_size + 1),
            nn.Conv3d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size-1)*(d_list[3]-1))//2,
                      dilation=d_list[3], groups=group_size + 1)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm3d(4 * (group_size + 1)),
            nn.Conv3d(4 * (group_size + 1), dim_xl, 1)
        )
    
    def forward(self, xh, xl, mask=None):
        xl = F.interpolate(xl, size=xh.shape[2:], mode='trilinear', align_corners=True)
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=xl.shape[2:], mode='trilinear', align_corners=True)
        xh_chunks = torch.chunk(xh, 4, dim=1)
        xl_chunks = torch.chunk(xl, 4, dim=1)
        group_size = self.dim_xl // 2
        mask = F.interpolate(mask, size=xl.shape[2:], mode='trilinear', align_corners=True).expand(-1, 1, -1, -1, -1) if mask else torch.zeros_like(xl_chunks[0][:, :1, :, :, :])
        
        x0 = self.g0(torch.cat((xh_chunks[0], xl_chunks[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh_chunks[1], xl_chunks[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh_chunks[2], xl_chunks[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh_chunks[3], xl_chunks[3], mask), dim=1))
        x = torch.cat((x0, x1, x2, x3), dim=1)
        return self.tail_conv(x)

# GroupedMultiAxisHadamardProductAttention3D module
class GroupedMultiAxisHadamardProductAttention3D(nn.Module):
    def __init__(self, dim_in, dim_out, small_d=4, small_h=8, small_w=8):
        super().__init__()
        c_dim_in = dim_in // 4
        k_size, pad = 3, (3 - 1) // 2
        
        self.params_hw = nn.Parameter(torch.ones(1, c_dim_in, small_h, small_w))
        self.conv_hw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1)
        )
        
        self.params_dh = nn.Parameter(torch.ones(1, c_dim_in, small_d, small_h))
        self.conv_dh = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1)
        )
        
        self.params_dw = nn.Parameter(torch.ones(1, c_dim_in, small_d, small_w))
        self.conv_dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1)
        )
        
        self.dw = nn.Sequential(
            nn.Conv3d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv3d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )
        
        self.norm1 = LayerNorm3d(dim_in)
        self.norm2 = LayerNorm3d(dim_in)
        self.ldw = nn.Sequential(
            nn.Conv3d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv3d(dim_in, dim_out, 1)
        )
    
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, D, H, W = x1.size()
        
        att_hw = self.conv_hw(F.interpolate(self.params_hw, size=(H, W), mode='bilinear', align_corners=True)).unsqueeze(2).expand(-1, -1, D, -1, -1)
        att_dh = self.conv_dh(F.interpolate(self.params_dh, size=(D, H), mode='bilinear', align_corners=True)).unsqueeze(4).expand(-1, -1, -1, -1, W)
        att_dw = self.conv_dw(F.interpolate(self.params_dw, size=(D, W), mode='bilinear', align_corners=True)).unsqueeze(3).expand(-1, -1, -1, H, -1)
        
        x1 *= att_hw
        x2 *= att_dh
        x3 *= att_dw
        x4 = self.dw(x4)
        
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.ldw(self.norm2(x))

# DoubleConv3D module
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if not mid_channels else mid_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

# UpBlock3D module
class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_ghpa=False):
        super().__init__()
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.gab = GroupAggregationBridge3D(out_channels, out_channels)
        self.block = GroupedMultiAxisHadamardProductAttention3D(out_channels, out_channels) if use_ghpa else DoubleConv3D(out_channels, out_channels)
    
    def forward(self, x_skip, x_up):
        x_up = self.up_conv(F.interpolate(x_up, scale_factor=2, mode='trilinear', align_corners=True))
        return self.block(self.gab(x_up, x_skip))

# LightGEUnet3D model
class LightEGEUnet3D(nn.Module):
    def __init__(self, in_chans, num_classes, c_list, bridge=True, drop_rate=0.0):
        super().__init__()
        self.in_chans, self.num_classes, self.c_list = in_chans, num_classes, c_list
        assert len(c_list) == 6, "c_list requires 6 elements"
        
        self.inc = DoubleConv3D(in_chans, c_list[0])
        self.down1 = nn.Sequential(nn.Conv3d(c_list[0], c_list[1], 2, 2), DoubleConv3D(c_list[1], c_list[1]))
        self.down2 = nn.Sequential(nn.Conv3d(c_list[1], c_list[2], 2, 2), DoubleConv3D(c_list[2], c_list[2]))
        self.down3 = nn.Sequential(nn.Conv3d(c_list[2], c_list[3], 2, 2), GroupedMultiAxisHadamardProductAttention3D(c_list[3], c_list[3]))
        self.down4 = nn.Sequential(nn.Conv3d(c_list[3], c_list[4], 2, 2), GroupedMultiAxisHadamardProductAttention3D(c_list[4], c_list[4]))
        self.bottleneck = GroupedMultiAxisHadamardProductAttention3D(c_list[4], c_list[5])
        
        self.up1 = UpBlock3D(c_list[5], c_list[4], True)
        self.up2 = UpBlock3D(c_list[4], c_list[3], True)
        self.up3 = UpBlock3D(c_list[3], c_list[2], False)
        self.up4 = UpBlock3D(c_list[2], c_list[1], False)
        
        self.outc = nn.Conv3d(c_list[1], num_classes, 1)
        self.dropout = nn.Dropout3d(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.bottleneck(x5)
        
        x = self.up1(x5, x6)
        x = self.up2(x4, x)
        x = self.up3(x3, x)
        x = self.up4(x2, x)
        
        return self.outc(self.dropout(x))
