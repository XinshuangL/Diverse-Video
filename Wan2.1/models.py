import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, remove_spectral_norm

# latent shape: torch.Size([16, 21, 60, 106])

class ResidualTanh(nn.Module):
    def forward(self, x):
        return x + 0.5 * torch.tanh(x)

def ConvBlock2D(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), use_activation=True):
    return nn.Sequential(
        spectral_norm(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride
            )
        ),
        ResidualTanh() if use_activation else nn.Identity()
    )

def ConvBlock3D(input_channels, output_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1), use_activation=True):
    return nn.Sequential(
        spectral_norm(
            nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride
            )
        ),
        ResidualTanh() if use_activation else nn.Identity()
    )

def TransposedConvBlock2D(input_channels, output_channels, kernel_size=(2, 2), stride=(2, 2), use_activation=True):
    return nn.Sequential(
        spectral_norm(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride
            )
        ),
        ResidualTanh() if use_activation else nn.Identity()
    )

def LinearBlock(input_channels, output_channels, use_activation=True):
    return nn.Sequential(
        spectral_norm(
            nn.Linear(input_channels, output_channels)
        ),
        ResidualTanh() if use_activation else nn.Identity()
    )

def remove_sn_from_model(model: nn.Module) -> nn.Module:
    model.eval()
    for m in model.modules():
        for pname in list(m._parameters):
            if pname.endswith("_orig"):
                remove_spectral_norm(m, pname[:-5])  # e.g. weight_orig -> weight
    return model

class LatentInterpolation(nn.Module):
    def __init__(self, latent_dim=16):
        super(LatentInterpolation, self).__init__()
        self.H = 72
        self.W = 128

        self.down_1 = ConvBlock2D(latent_dim, latent_dim*2, stride=2)
        self.down_2 = ConvBlock2D(latent_dim*2, latent_dim*4, stride=2)   
        self.down_3 = ConvBlock2D(latent_dim*4, latent_dim*8, stride=2)

        self.up_1 = TransposedConvBlock2D(latent_dim*(8+8), latent_dim*4, stride=2)
        self.up_2 = TransposedConvBlock2D(latent_dim*(4+4+4), latent_dim*2, stride=2)
        self.up_3 = TransposedConvBlock2D(latent_dim*(2+2+2), latent_dim*1, stride=2)

        self.final_conv = ConvBlock2D(latent_dim, latent_dim, stride=1, use_activation=False)

    def forward(self, latent_before, latent_after):
        # input shape: N,C,H,W
        # output shape: N,C,H,W
        N = latent_before.shape[0]
        assert N == latent_after.shape[0]
        original_H, original_W = latent_before.shape[2:]

        mean_latent = (latent_before + latent_after) / 2
        cat_latent = torch.cat([latent_before, latent_after], dim=0)

        latent_cat_interpolated = F.interpolate(cat_latent, size=(self.H, self.W), mode='bilinear')

        # downsampling
        cat_2x = self.down_1(latent_cat_interpolated)
        cat_4x = self.down_2(cat_2x)
        cat_8x = self.down_3(cat_4x)

        # split
        before_2x = cat_2x[:N, :, :, :]
        after_2x = cat_2x[N:, :, :, :]
        before_4x = cat_4x[:N, :, :, :]
        after_4x = cat_4x[N:, :, :, :]
        before_8x = cat_8x[:N, :, :, :]
        after_8x = cat_8x[N:, :, :, :]

        # upsampling
        up_4x = self.up_1(torch.cat([before_8x, after_8x], dim=1))
        up_2x = self.up_2(torch.cat([up_4x, before_4x, after_4x], dim=1))
        up_1x = self.up_3(torch.cat([up_2x, before_2x, after_2x], dim=1))

        # final output
        residual = F.interpolate(up_1x, size=(original_H, original_W), mode='bilinear')
        residual = self.final_conv(residual)

        return mean_latent + residual, residual

################
# code for embed
################
from torchvision.transforms.functional import gaussian_blur

class FrameEmbedModel(nn.Module):
    def __init__(self, latent_dim=16, embed_dim=16):
        super(FrameEmbedModel, self).__init__()
        self.latent_dim = latent_dim
        self.down = ConvBlock2D(latent_dim, latent_dim, stride=1)
        self.mid1 = ConvBlock2D(latent_dim, latent_dim, stride=1)   
        self.mid2 = ConvBlock2D(latent_dim, latent_dim, stride=1)   
        self.final_linear = LinearBlock(latent_dim, embed_dim, use_activation=False)

    def forward(self, latent):
        # input shape: N,C,H,W
        N, C, H, W = latent.shape
        assert H == 60 and W == 104

        # k = int(2 * math.ceil(3 * sigma) + 1)
        latent_blurred = gaussian_blur(latent, kernel_size=31, sigma=5.0)
        latent_pooled = F.avg_pool2d(
            latent_blurred,
            kernel_size=(14, 14),
            stride=(10, 10),
            padding=(2, 0),
            count_include_pad=False
        )
        assert latent_pooled.shape == (N, C, 6, 10)

        # downsampling
        x_down = self.down(latent_pooled)
        x_down = F.avg_pool2d(x_down, kernel_size=(2, 2), stride=(2, 2)) # 3 x 5
        x_mid = self.mid1(x_down)
        x_mid = self.mid2(x_mid)
        x_linear = F.adaptive_avg_pool2d(x_mid, (1, 1)).view(N, -1)

        # output
        out = self.final_linear(x_linear)
        out = F.normalize(out, p=2, dim=-1)
        return out

class VideoEmbedModel(nn.Module):
    def __init__(self, latent_dim: int = 16, embed_dim: int = 16):
        super(VideoEmbedModel, self).__init__()

        self.down = ConvBlock3D(latent_dim, latent_dim, kernel_size=(2, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1))
        self.mid1 = ConvBlock2D(latent_dim, latent_dim, stride=1)
        self.mid2 = ConvBlock2D(latent_dim, latent_dim, stride=1)
        self.final_linear = LinearBlock(latent_dim, embed_dim, use_activation=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent[:, :, 1:].contiguous()
        N, C, T, H, W = latent.shape
        assert T == 20 and H == 60 and W == 104

        # k = int(2 * math.ceil(3 * sigma) + 1)
        latent_NTCHW = latent.permute(0, 2, 1, 3, 4).contiguous().view(N*T, C, H, W)
        latent_blurred = gaussian_blur(latent_NTCHW, kernel_size=31, sigma=5.0)
        latent_blurred = latent_blurred.view(N, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        latent_pooled = F.avg_pool3d(
            latent_blurred,
            kernel_size=(14, 14, 14),
            stride=(10, 10, 10),
            padding=(2, 2, 0),
            count_include_pad=False
        )
        assert latent_pooled.shape == (N, C, 2, 6, 10)

        x_down = self.down(latent_pooled)
        x_down = x_down.view(N, C, 6, 10)
        x_down = F.avg_pool2d(x_down, kernel_size=(2, 2), stride=(2, 2)) # 3 x 5
        x_mid = self.mid1(x_down)
        x_mid = self.mid2(x_mid)
        x_linear = F.adaptive_avg_pool2d(x_mid, (1, 1)).view(N, -1)

        # output
        out = self.final_linear(x_linear)
        out = F.normalize(out, p=2, dim=-1)
        return out
