"""Small-image residual networks; width is the number of stem channels."""
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(y)) + self.shortcut(x))


class SmallResNet(nn.Module):
    def __init__(self, width=8, classes=4, blocks=(1, 1, 1, 1)):
        super().__init__()
        layers = [nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU()]
        previous = width
        for stage, count in enumerate(blocks):
            channels = width * 2**stage
            for index in range(count):
                stride = 2 if stage and index == 0 else 1
                layers.append(BasicBlock(previous, channels, stride))
                previous = channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(previous, classes)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


class ViTBlock(nn.Module):
    """Pre-norm multihead attention and a GELU MLP, with no dropout."""
    def __init__(self, dimension, heads=2, mlp_ratio=2):
        super().__init__()
        assert dimension % heads == 0
        self.heads = heads
        self.scale = (dimension // heads) ** -.5
        self.norm1 = nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.proj = nn.Linear(dimension, dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension * mlp_ratio), nn.GELU(),
                                 nn.Linear(dimension * mlp_ratio, dimension))

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(out)
        return x + self.mlp(self.norm2(x))


class SmallViT(nn.Module):
    """ViT with learned CLS/position tokens. Width is token dimension."""
    def __init__(self, width=24, classes=4, depth=4, heads=2, patch_size=4, mlp_ratio=2, image_size=32):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch = nn.Conv2d(3, width, patch_size, patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, (image_size // patch_size)**2 + 1, width))
        self.blocks = nn.Sequential(*[ViTBlock(width, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)
        nn.init.trunc_normal_(self.cls, std=.02)
        nn.init.trunc_normal_(self.position, std=.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=.02)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls.expand(x.shape[0], -1, -1), tokens), dim=1)
        return self.head(self.norm(self.blocks(tokens + self.position))[:, 0])


def build_model(config, width):
    if config['architecture'] == 'SmallViT':
        return SmallViT(width, depth=config['depth'], heads=config['heads'],
                        patch_size=config['patch_size'], mlp_ratio=config['mlp_ratio'],
                        image_size=config.get('image_size', 32))
    if config['architecture'] == 'SmallResNet-10':
        return SmallResNet(width, blocks=tuple(config['blocks']))
    raise ValueError(f"Unknown architecture: {config['architecture']}")
