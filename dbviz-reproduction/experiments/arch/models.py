"""dbViz CIFAR classifiers; model checkpoints are plain PyTorch state dictionaries."""
from pathlib import Path

ARCHITECTURES = ("fc", "mixer", "vit", "vgg", "dense", "resnet", "wrn10")


def create(name):
    if name == "fc":
        from .source.fcnet import FCNet
        return FCNet()
    if name == "mixer":
        from .source.mlpmixer import MLPMixer4
        return MLPMixer4()
    if name == "vit":
        from .source.vit import ViT4
        return ViT4()
    if name == "vgg":
        from .source.vgg import VGG
        return VGG("VGG19")
    if name == "dense":
        from .source.densenet import DenseNet121
        return DenseNet121()
    if name == "resnet":
        from .source.resnet import ResNet18
        return ResNet18()
    if name == "wrn10":
        from .source.wideresnet import WideResNet
        return WideResNet(depth=28, num_classes=10, widen_factor=10)
    raise ValueError(f"Unknown architecture {name!r}")


def checkpoint(work, name, seed):
    return Path(work) / "checkpoints" / name / f"seed{seed}.pth"
