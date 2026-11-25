"""Download CIFAR-10 using TorchVision, or verify an existing copy."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    from torchvision.datasets import CIFAR10

    for split, train, expected in [("train", True, 50000), ("test", False, 10000)]:
        dataset = CIFAR10(root=args.data, train=train, download=not args.verify_only)
        if len(dataset) != expected:
            raise ValueError(f"Unexpected number of {split} images: {len(dataset)}")
        print(f"{split}: {len(dataset)} images, verified in {args.data.resolve()}")


if __name__ == "__main__":
    main()
