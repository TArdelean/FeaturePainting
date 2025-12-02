import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    r"""
    From: https://github.com/facebookresearch/ConvNeXt/tree/main
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ContrastNet(nn.Module):
    def __init__(self, latent=32):
        super(ContrastNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            LayerNorm(512, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            LayerNorm(512, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(512, 128, 1),
        )

        self.proj_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

    def emb(self, x):
        return self.net(x)

    def project(self, embedding):
        return self.proj_layer(embedding.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x, fun):
        if fun == 'emb':
            return self.emb(x)
        elif fun == 'proj':
            return self.project(self.emb(x))
        else:
            raise Exception(f"Undefined function {fun}")
