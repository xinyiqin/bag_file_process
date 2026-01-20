# import torch
# import torch.nn as nn
# import torch.fft
#
# # Complementary Advantages:Exploiting Cross-Field FrequencyCorrelation for NIR-Assisted Image Denoising
# class FEFM(nn.Module):
#     def __init__(self, channels):
#         """
#         Frequency Exhaustive Fusion Mechanism (FEFM)
#         Args:
#             channels: 输入特征图的通道数
#         """
#         super().__init__()
#
#         # 点卷积和深度卷积层定义
#         self.point_conv_Q = nn.Conv2d(channels, channels, kernel_size=1)
#         self.depth_conv_Q = nn.Conv2d(channels, channels, kernel_size=3,
#                                       padding=1, groups=channels)
#
#         self.point_conv_K = nn.Conv2d(channels, channels, kernel_size=1)
#         self.depth_conv_K = nn.Conv2d(channels, channels, kernel_size=3,
#                                       padding=1, groups=channels)
#
#         self.point_conv_V = nn.Conv2d(channels, channels, kernel_size=1)
#         self.depth_conv_V = nn.Conv2d(channels, channels, kernel_size=3,
#                                       padding=1, groups=channels)
#
#         # 可学习参数
#         self.alpha = nn.Parameter(torch.tensor(1.0))  # CFR缩放因子
#         self.lambd = nn.Parameter(torch.tensor(0.5))  # DFR权重因子
#
#         # 初始化参数
#         for conv in [self.point_conv_Q, self.point_conv_K, self.point_conv_V,
#                      self.depth_conv_Q, self.depth_conv_K, self.depth_conv_V]:
#             nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
#             if conv.bias is not None:
#                 nn.init.constant_(conv.bias, 0)
#
#     def forward(self, F_R, F_N):
#         """
#         前向传播
#         Args:
#             F_R: RGB特征图 [B, C, H, W]
#             F_N: NIR特征图 [B, C, H, W]
#         Returns:
#             融合后的特征图 [B, C, H, W]
#         """
#         # ===== 1. 特征编码 =====
#         Q = self.depth_conv_Q(self.point_conv_Q(F_R))
#         K = self.depth_conv_K(self.point_conv_K(F_N))
#         V = self.depth_conv_V(self.point_conv_V(F_N))
#
#         # ===== 2. 频域转换 =====
#         F_Q = torch.fft.fft2(Q, dim=(-2, -1))
#         F_K = torch.fft.fft2(K, dim=(-2, -1))
#
#         # ===== 3. Common Feature Reinforcement (CFR) =====
#         # 逐元素乘积
#         elem_product = F_Q * F_K
#
#         # 矩阵乘法计算注意力
#         B, C, H, W = F_Q.shape
#         F_Q_flat = F_Q.view(B, C, -1)  # [B, C, H*W]
#         F_K_flat = F_K.view(B, C, -1)  # [B, C, H*W]
#
#         # 计算注意力权重 (使用幅度)
#         attn_matrix = torch.matmul(F_Q_flat, F_K_flat.transpose(1, 2))  # [B, C, C]
#         attn_weights = torch.softmax(attn_matrix.abs() / self.alpha, dim=-1)  # [B, C, C]
#
#         # 将attn_weights转换为复数类型
#         attn_weights_complex = torch.complex(attn_weights, torch.zeros_like(attn_weights))
#
#         # 应用注意力权重
#         elem_product_flat = elem_product.view(B, C, -1)  # [B, C, H*W]
#         F_CFR_flat = torch.matmul(attn_weights_complex, elem_product_flat)  # [B, C, H*W]
#         F_CFR = F_CFR_flat.view(B, C, H, W)  # [B, C, H, W]
#
#         # 逆FFT转换回空间域
#         cfr_spatial = torch.fft.ifft2(F_CFR, dim=(-2, -1)).real
#
#         # ===== 4. Differential Feature Reinforcement (DFR) =====
#         F_DFR = V - self.lambd * V * cfr_spatial
#
#         # ===== 5. 最终融合 =====
#         output = Q * cfr_spatial + F_DFR
#
#         return output
#
#
# if __name__ == "__main__":
#     # 测试配置
#     batch_size = 4
#     channels = 64
#     height = 128
#     width = 128
#
#     # 创建FEFM模块
#     fefm = FEFM(channels)
#
#     # 创建模拟输入特征图
#     F_R = torch.randn(batch_size, channels, height, width)  # RGB特征
#     F_N = torch.randn(batch_size, channels, height, width)  # NIR特征
#
#     print("输入特征图尺寸:")
#     print(f"F_R: {F_R.shape}")
#     print(f"F_N: {F_N.shape}")
#
#     # 前向传播
#     output = fefm(F_R, F_N)
#
#     print("\n输出特征图尺寸:", output.shape)
#


import torch
import torch.nn as nn
import torch.fft


class FEFM(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, reduction=8):
        """
        Frequency Exhaustive Fusion Mechanism (FEFM) with support for different channel inputs
        Args:
            in_channels1: 第一个输入特征图的通道数
            in_channels2: 第二个输入特征图的通道数
            out_channels: 输出特征图的通道数
            reduction: 注意力机制中的通道缩减比例
        """
        super().__init__()

        # 通道调整层，将不同通道数的输入调整到统一的中间通道数
        mid_channels = max(in_channels1, in_channels2)
        self.conv_r = nn.Conv2d(in_channels1, mid_channels, kernel_size=1)
        self.conv_n = nn.Conv2d(in_channels2, mid_channels, kernel_size=1)

        # 点卷积和深度卷积层定义
        self.point_conv_Q = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.depth_conv_Q = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                      padding=1, groups=mid_channels)

        self.point_conv_K = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.depth_conv_K = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                      padding=1, groups=mid_channels)

        self.point_conv_V = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.depth_conv_V = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                      padding=1, groups=mid_channels)

        # 最终输出层，调整通道数到out_channels
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # 注意力机制参数
        self.alpha = nn.Parameter(torch.tensor(1.0))  # CFR缩放因子
        self.lambd = nn.Parameter(torch.tensor(0.5))  # DFR权重因子

        # 自适应特征融合参数
        self.beta = nn.Parameter(torch.tensor(0.5))  # 频域融合与空间融合的平衡因子

        # 通道注意力机制，用于自适应融合不同来源的特征
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        d = max(int(mid_channels / reduction), 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(mid_channels, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, mid_channels * 2, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, in_feats):
        """
        前向传播（处理不同通道数的输入特征）
        Args:
            F_R: 第一个特征图 [B, C1, H, W]
            F_N: 第二个特征图 [B, C2, H, W]
        Returns:
            融合后的特征图 [B, out_channels, H, W]
        """
        F_R, F_N = in_feats[0], in_feats[1]
        # 调整通道数
        F_R = self.conv_r(F_R)
        F_N = self.conv_n(F_N)

        # ===== 1. 特征拼接 =====
        F_concat = torch.cat([F_R, F_N], dim=1)  # 拼接后的通道数为2*mid_channels

        # ===== 2. 通道注意力，自适应融合两个特征 =====
        B, C_total, H, W = F_concat.shape
        F_concat_reshaped = F_concat.view(B, 2, C_total // 2, H, W)  # [B, 2, mid_channels, H, W]
        feats_sum = torch.sum(F_concat_reshaped, dim=1)  # [B, mid_channels, H, W]

        # 生成注意力权重
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, 2, C_total // 2, 1, 1))

        # 应用注意力权重
        F_weighted = torch.sum(F_concat_reshaped * attn, dim=1)  # [B, mid_channels, H, W]

        # ===== 3. 特征编码 =====
        Q = self.depth_conv_Q(self.point_conv_Q(F_weighted))
        K = self.depth_conv_K(self.point_conv_K(F_weighted))
        V = self.depth_conv_V(self.point_conv_V(F_weighted))

        # ===== 4. 频域转换 =====
        # Convert to float32 for FFT to avoid cuFFT power-of-two restriction in fp16
        # Keep results as complex float32 throughout frequency domain operations
        original_dtype = Q.dtype
        Q_fp32 = Q.to(torch.float32)
        K_fp32 = K.to(torch.float32)
        F_Q = torch.fft.fft2(Q_fp32, dim=(-2, -1))  # Keep as complex float32
        F_K = torch.fft.fft2(K_fp32, dim=(-2, -1))  # Keep as complex float32

        # ===== 5. Common Feature Reinforcement (CFR) =====
        elem_product = F_Q * F_K

        # 计算注意力权重
        B, C, H, W = F_Q.shape
        F_Q_flat = F_Q.view(B, C, -1)  # [B, mid_channels, H*W]
        F_K_flat = F_K.view(B, C, -1)  # [B, mid_channels, H*W]

        attn_matrix = torch.matmul(F_Q_flat, F_K_flat.transpose(1, 2))  # [B, mid_channels, mid_channels]
        attn_weights = torch.softmax(attn_matrix.abs() / self.alpha, dim=-1)

        # 应用注意力权重
        # Convert attention weights to complex float32 to match FFT output dtype
        attn_weights_fp32 = attn_weights.to(torch.float32)
        attn_weights_complex = torch.complex(attn_weights_fp32, torch.zeros_like(attn_weights_fp32))
        elem_product_flat = elem_product.view(B, C, -1)
        F_CFR_flat = torch.matmul(attn_weights_complex, elem_product_flat)
        F_CFR = F_CFR_flat.view(B, C, H, W)

        # 逆FFT转换回空间域
        # IFFT on complex float32, then convert real part to original dtype
        cfr_spatial = torch.fft.ifft2(F_CFR, dim=(-2, -1)).real.to(original_dtype)

        # ===== 6. Differential Feature Reinforcement (DFR) =====
        F_DFR = V - self.lambd * V * cfr_spatial

        # ===== 7. 最终融合 =====
        # 结合频域融合结果和空间域特征
        freq_output = Q * cfr_spatial + F_DFR
        output = self.beta * freq_output + (1 - self.beta) * F_weighted

        # 调整输出通道数到out_channels
        output = self.out_conv(output)

        return output


if __name__ == "__main__":
    # 测试配置 - 不同通道数的输入
    batch_size = 4
    height = 128
    width = 128

    # 创建FEFM模块，指定三个通道数参数
    in_channels1 = 32  # 第一个特征图的通道数
    in_channels2 = 64  # 第二个特征图的通道数
    out_channels = 96  # 输出特征图的通道数

    fefm = FEFM(in_channels1, in_channels2, out_channels)

    # 创建模拟输入特征图 - 不同通道数
    F_R = torch.randn(batch_size, in_channels1, height, width)
    F_N = torch.randn(batch_size, in_channels2, height, width)

    print("输入特征图尺寸:")
    print(f"F_R: {F_R.shape}")
    print(f"F_N: {F_N.shape}")

    # 前向传播
    output = fefm(F_R, F_N)

    print("\n输出特征图尺寸:", output.shape)
    # 验证输出通道数是否为指定的out_channels
    print(f"输出通道数: {output.shape[1]} (应为{out_channels})")
