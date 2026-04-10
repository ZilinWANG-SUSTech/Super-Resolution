import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from utils import NETWORK_REGISTRY
import math

# ==============================================================================
# Utils.py Equivalents (Strictly translated from TF logic)
# ==============================================================================

def diff_x(input, r):
    # PyTorch spatial dims are 2 (H) and 3 (W). TF diff_x operates on H (axis=2).
    left = input[:, :, r:2 * r + 1, :]
    middle = input[:, :, 2 * r + 1:, :] - input[:, :, :-2 * r - 1, :]
    right = input[:, :, -1:, :] - input[:, :, -2 * r - 1:-r - 1, :]
    return torch.cat([left, middle, right], dim=2)

def diff_y(input, r):
    # TF diff_y operates on W (axis=3).
    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:-r - 1]
    return torch.cat([left, middle, right], dim=3)

def box_filter(x, r):
    # Cumulative sum over H (dim 2) and W (dim 3)
    return diff_y(torch.cumsum(diff_x(torch.cumsum(x, dim=2), r), dim=3), r)

def guided_filter(x, y, r, eps=1e-8):
    # x, y: [B, C, H, W]. r: radius
    N = box_filter(torch.ones_like(x), r)

    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    var_x = box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b
    return output

def bilinear_sampler_1d_h(input_images, x_offset):
    # Strictly implements horizontal shift
    # x_offset is relative to full width in TF code, mapped to grid_sample [-1, 1] range
    B, C, H, W = input_images.shape
    device = input_images.device

    # 1. 添加 indexing='ij'，消除 PyTorch 的 UserWarning
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    grid_x = grid_x.to(device).unsqueeze(0).repeat(B, 1, 1)
    grid_y = grid_y.to(device).unsqueeze(0).repeat(B, 1, 1)

    # ================= 核心修复 =================
    # 如果视差图有多个通道 (因为 Pol-SAR 是3通道引导滤波的)，
    # 求平均将其压缩回单通道 (1D Spatial Displacement)
    if x_offset.dim() == 4 and x_offset.shape[1] > 1:
        x_offset = x_offset.mean(dim=1, keepdim=True)
    # ============================================

    # Offset multiplier to match TF's x_t_flat + x_offset * _width_f
    # 此时 x_offset 必定为 [B, 1, H, W], squeeze(1) 后变为 [B, H, W], 与 grid_x 完美广播匹配
    grid_x = grid_x + x_offset.squeeze(1) * 2.0

    new_grid = torch.stack([grid_x, grid_y], dim=-1)
    # TF wrap_mode='border' corresponds to padding_mode='border'
    output = F.grid_sample(input_images, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return output


# ==============================================================================
# Ops.py & inferred Layers.py Equivalents
# ==============================================================================

class OpsConv(nn.Module):
    """ Equivalent to ops.conv in TF code """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, sn=False):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=True)
        # Apply Spectral Normalization if sn=True
        self.conv = spectral_norm(conv) if sn else conv

    def forward(self, x):
        return self.conv(x)

class GeneralConv2d(nn.Module):
    """ Inferred from layers.general_conv2d calls in model.py """
    def __init__(self, in_channels, out_channels, kernel_size, rate=1, do_norm=False, do_relu=False):
        super().__init__()
        # SAME padding calculation with dilation (rate)
        padding = (kernel_size - 1) * rate // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=rate, bias=not do_norm)
        
        self.norm = nn.BatchNorm2d(out_channels) if do_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if do_relu else nn.Identity()
        
        nn.init.trunc_normal_(self.conv.weight, std=0.02)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


# ==============================================================================
# Model.py Basic Blocks
# ==============================================================================

class PreactConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=True)
        nn.init.trunc_normal_(self.conv.weight, std=0.02)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(self.relu(x))

class DenseBlock(nn.Module):
    def __init__(self, in_channels, n_layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(n_layers):
            self.layers.append(PreactConv(current_channels, growth_rate))
            current_channels += growth_rate
        self.out_channels = current_channels

    def forward(self, x):
        new_features = []
        for layer in self.layers:
            out = layer(x)
            new_features.append(out)
            x = torch.cat([x, out], dim=1) # Concat along channels
        return x, torch.cat(new_features, dim=1)

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        nn.init.trunc_normal_(self.conv.weight, std=0.02)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(self.relu(x))

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        nn.init.trunc_normal_(self.upconv.weight, std=0.02)
        nn.init.constant_(self.upconv.bias, 0.0)

    def forward(self, block_to_upsample, skip_connection):
        l = self.upconv(block_to_upsample)
        if l.shape[2:] != skip_connection.shape[2:]:
            l = F.interpolate(l, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([l, skip_connection], dim=1)

class GoogleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sn = True
        self.f_conv = OpsConv(channels, channels // 8, kernel_size=1, sn=sn)
        self.g_conv = OpsConv(channels, channels // 8, kernel_size=1, sn=sn)
        self.h_conv = OpsConv(channels, channels // 2, kernel_size=1, sn=sn)
        self.attn_conv = OpsConv(channels // 2, channels, kernel_size=1, sn=sn)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        f = self.max_pool(self.f_conv(x)).view(B, C // 8, -1)
        g = self.g_conv(x).view(B, C // 8, -1)
        h = self.max_pool(self.h_conv(x)).view(B, C // 2, -1)

        s = torch.bmm(g.transpose(1, 2), f)
        beta = F.softmax(s, dim=-1)

        o = torch.bmm(h, beta.transpose(1, 2)).view(B, C // 2, H, W)
        o = self.attn_conv(o)
        return self.gamma * o + x


# ==============================================================================
# Main Model Assembly (Strictly referencing `generator` in model.py)
# ==============================================================================

@NETWORK_REGISTRY.register()
class UGSRGenerator(nn.Module):
    def __init__(self, in_channels=3, guide_channels=3, scale_factor=4):
        """
        in_channels: 3 for Pol-SAR
        guide_channels: 3 for RGB
        scale_factor: Target upsampling scale (must be a power of 2, e.g., 4, 8, 16).
        """
        super().__init__()
        assert (scale_factor & (scale_factor - 1) == 0) and scale_factor > 1, "scale_factor must be a power of 2 (e.g., 2, 4, 8, 16)"
        
        self.scale_factor = scale_factor
        self.n_pool = int(math.log2(scale_factor))
        self.growth_rate = 32
        self.n_layers_per_block = [2] * self.n_pool + [3] + [3] * self.n_pool
        ngf = 32
        ndf = 64
        
        self.disp_range = 0.4
        self.disp_channel = 200
        self.guide_weight = 5

        # 1. DISP Network (Warping Cost Volume logic)
        self.disp_g1 = GeneralConv2d(in_channels + self.disp_channel, ngf, kernel_size=5)
        self.disp_g2 = GeneralConv2d(ngf, ndf * 2, kernel_size=3, do_norm=True)
        self.disp_g3 = GeneralConv2d(ndf * 2, ndf * 2, kernel_size=3, do_norm=True)
        self.disp_g4 = GeneralConv2d(ndf * 2, ndf * 2, kernel_size=3, do_norm=True, rate=2)
        self.disp_g5 = GeneralConv2d(ndf * 2, ndf, kernel_size=3, do_norm=True, rate=2)
        
        self.disp_db1 = DenseBlock(ndf, 3, self.growth_rate)
        self.disp_db2 = DenseBlock(self.disp_db1.out_channels, 3, self.growth_rate)
        self.disp_db3 = DenseBlock(self.disp_db2.out_channels, 3, self.growth_rate)
        self.disp_db4 = DenseBlock(self.disp_db3.out_channels, 3, self.growth_rate)
        
        self.disp_g6 = GeneralConv2d(self.disp_db4.out_channels, self.disp_channel, kernel_size=3, do_norm=True)

        # 2. Guide Branch
        self.guide_dbs = nn.ModuleList()
        self.guide_tds = nn.ModuleList()
        g_ch = guide_channels # Strictly warps the first channel in original, but allows full here
        for i in range(self.n_pool):
            db = DenseBlock(g_ch, self.n_layers_per_block[i], self.growth_rate)
            self.guide_dbs.append(db)
            g_ch = db.out_channels
            self.guide_tds.append(TransitionDown(g_ch, ngf + self.growth_rate * sum(self.n_layers_per_block[:i+1])))
            g_ch = ngf + self.growth_rate * sum(self.n_layers_per_block[:i+1])
        
        # 3. Thermal/Pol-SAR Branch
        self.lr_c1 = GeneralConv2d(in_channels, ngf, kernel_size=7)
        self.lr_c2 = GeneralConv2d(ngf, ngf * 2, kernel_size=3)
        self.lr_c3 = GeneralConv2d(ngf * 2, ngf * 2, kernel_size=3)
        
        self.lr_dbs = nn.ModuleList()
        self.lr_preacts = nn.ModuleList()
        lr_ch = ngf * 2

        calculated_skip_channels = []
        temp_lr_ch = lr_ch
        for i in range(self.n_pool):
            db_out_ch = temp_lr_ch + self.n_layers_per_block[i] * self.growth_rate
            calculated_skip_channels.append(db_out_ch)
            temp_lr_ch = ngf + self.growth_rate * sum(self.n_layers_per_block[:i+1])
        calculated_skip_channels = calculated_skip_channels[::-1]

        for i in range(self.n_pool):
            db = DenseBlock(lr_ch, self.n_layers_per_block[i], self.growth_rate)
            self.lr_dbs.append(db)
            lr_ch = db.out_channels
            n_filters_i = ngf + self.growth_rate * sum(self.n_layers_per_block[:i+1])
            self.lr_preacts.append(PreactConv(lr_ch, n_filters_i, kernel_size=1))
            lr_ch = n_filters_i

        # 4. Attention
        self.attn_guide = GoogleAttention(g_ch)
        self.attn_ther = GoogleAttention(lr_ch)

        # 5. Decoder
        self.bottle_db = DenseBlock(g_ch, self.n_layers_per_block[self.n_pool], self.growth_rate)
        dec_ch = self.bottle_db.out_channels
        
        self.up_tus = nn.ModuleList()
        self.up_dbs = nn.ModuleList()
        for i in range(self.n_pool):
            n_filters_keep = self.growth_rate * self.n_layers_per_block[self.n_pool + i]
            
            if i == 0:
                # For the first TransitionUp, the input comes from the bottleneck block
                trans_in_ch = self.n_layers_per_block[self.n_pool] * self.growth_rate
            else:
                # For subsequent TransitionUps, the input comes from the previous up_db DenseBlock
                trans_in_ch = self.n_layers_per_block[self.n_pool + i] * self.growth_rate
            
            self.up_tus.append(TransitionUp(trans_in_ch, n_filters_keep))

            # skip_ch = ngf * 2 if i == (self.n_pool - 1) else (ngf * 2 + self.growth_rate * sum(self.n_layers_per_block[:self.n_pool - 1 - i]))
            # if i == (self.n_pool - 1): 
            #     skip_ch = ngf * 2
            # else:
            #     skip_ch = ngf * 2 + self.growth_rate * sum(self.n_layers_per_block[:self.n_pool - 1 - i])
            skip_ch = calculated_skip_channels[i]

            dec_ch = n_filters_keep + skip_ch
            
            db = DenseBlock(dec_ch, self.n_layers_per_block[self.n_pool + i + 1], self.growth_rate)
            self.up_dbs.append(db)
            dec_ch = db.out_channels

        self.c6 = GeneralConv2d(dec_ch, in_channels, kernel_size=3)

    def forward(self, inputA, guide):
        B, C_l, H_l, W_l = inputA.shape
        _, C_g, H_g, W_g = guide.shape

        # --- 1. Disparity Estimation & Warping ---
        # Extract first channel of guide for disparity calculation (as per TF: l = guide[:, :, :, 0:1])
        l = guide[:, 0:1, :, :]
        l = F.interpolate(l, size=(H_l, W_l), mode='bilinear', align_corners=True)
        
        delta = (2 * self.disp_range) / (self.disp_channel - 1)
        warped_guides = []
        for index in range(self.disp_channel):
            depth = -self.disp_range + delta * index
            disp_map = torch.ones(B, 1, H_l, W_l, device=inputA.device) * depth
            x = bilinear_sampler_1d_h(l, disp_map)
            warped_guides.append(x)
            
        disparity = torch.cat(warped_guides, dim=1) # [B, 200, H_l, W_l]
        input_disp = torch.cat([inputA, disparity], dim=1)

        stack_d = self.disp_g5(self.disp_g4(self.disp_g3(self.disp_g2(self.disp_g1(input_disp)))))
        stack_d, _ = self.disp_db1(stack_d)
        stack_d, _ = self.disp_db2(stack_d)
        stack_d, _ = self.disp_db3(stack_d)
        stack_d, _ = self.disp_db4(stack_d)
        
        o_g5 = F.softmax(self.disp_g6(stack_d), dim=1)
        confidence = torch.max(o_g5, dim=1)[0]
        
        beta_val = 10.0
        x_range = torch.arange(self.disp_channel, dtype=o_g5.dtype, device=inputA.device).view(1, -1, 1, 1)
        layer_disp = torch.sum(F.softmax(o_g5 * beta_val, dim=1) * x_range, dim=1, keepdim=True)
        
        refined_disp = ((layer_disp / (self.disp_channel + 1)) * 2 * self.disp_range) - self.disp_range
        filtered_refined_disp = guided_filter(inputA, refined_disp, self.guide_weight)
        filtered_refined_disp = F.interpolate(filtered_refined_disp, size=(H_g, W_g), mode='bilinear', align_corners=True)
        
        warped = bilinear_sampler_1d_h(guide, filtered_refined_disp)

        # --- 2. Dual Stream Feature Extraction ---
        stack1 = warped
        for i in range(self.n_pool):
            stack1, _ = self.guide_dbs[i](stack1)
            stack1 = self.guide_tds[i](stack1)

        o_c1 = self.lr_c1(inputA)
        o_c2 = self.lr_c2(o_c1)
        o_c3 = self.lr_c3(o_c2)
        stack2 = F.leaky_relu(o_c3, negative_slope=0.2)
        
        skip_connection_list = []
        for i in range(self.n_pool):
            stack2, _ = self.lr_dbs[i](stack2)
            skip_connection_list.append(stack2)
            stack2 = self.lr_preacts[i](stack2)

        # Adjust skips
        # ---------------------------------------------------------
        # Dynamic Skip Connection Interpolation
        # For n_pool=2:
        # i=0: factor = 2**(2-0) = 4.0
        # i=1: factor = 2**(2-1) = 2.0
        # For n_pool=4 (16x):
        # i=0 -> 16.0, i=1 -> 8.0, i=2 -> 4.0, i=3 -> 2.0
        # ---------------------------------------------------------
        for i in range(self.n_pool):
            factor = float(2 ** (self.n_pool - i))
            skip_connection_list[i] = F.interpolate(
                skip_connection_list[i], 
                scale_factor=factor, 
                mode='bilinear', 
                align_corners=True
            )
        skip_connection_list = skip_connection_list[::-1]

        # --- 3. Attention Fusion ---
        stack1_befAt = torch.tanh(stack1)
        stack2_befAt = torch.tanh(stack2)
        
        stack1_affAt = self.attn_guide(stack1_befAt)
        stack2_affAt = self.attn_ther(stack2_befAt)
        
        stack = (stack1_affAt + stack2_affAt) / 2.0

        # --- 4. Decoder ---
        stack, block_to_upsample = self.bottle_db(stack)
        
        for i in range(self.n_pool):
            stack = self.up_tus[i](block_to_upsample, skip_connection_list[i])
            stack, block_to_upsample = self.up_dbs[i](stack)

        net = torch.tanh(self.c6(stack))
        output = (net + F.interpolate(inputA, size=(H_g, W_g), mode='bilinear', align_corners=True)) / 2.0
        
        return output # Returns just the final image, you can return a dict if needed for loss calc