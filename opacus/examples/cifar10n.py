#!/usr/bin/env python3

"""
Runs CIFAR-10 training with differential privacy, based on the MNIST example.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class CIFAR10ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch, stats):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        max_grad_norm = getattr(optimizer, "max_grad_norm", None)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(\u03b5 = {epsilon:.2f}, δ = {args.delta}) "
            f"Clip Norm: {max_grad_norm:.4f}" if max_grad_norm else ""
        )
        stats["epsilon"].append(epsilon)
        stats["clip_norm"].append(max_grad_norm)
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f}%)\n"
    )
    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Opacus CIFAR10 Example")

    parser.add_argument("--clipping", type=str, default="flat", choices=["flat", "per_layer", "adaptive", "dcsgdp", "dcsgde"])
    parser.add_argument("--percentile", type=float, default=0.3)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--bin-cnt", type=int, default=20)
    parser.add_argument("--histogram-std", type=float, default=6.0)
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-dp", action="store_true")
    parser.add_argument("--secure-rng", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--data-root", type=str, default="../cifar10")
    args = parser.parse_args()

    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_root, train=False, transform=transform),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    stats = {"clip_norm": [], "epsilon": []}
    model = CIFAR10ConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    privacy_engine = None
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        make_private_kwargs = {
            "module": model,
            "optimizer": optimizer,
            "data_loader": train_loader,
            "noise_multiplier": args.sigma,
            "max_grad_norm": args.max_per_sample_grad_norm,
            "clipping": args.clipping,
        }
        if args.clipping == "dcsgdp":
            make_private_kwargs.update({
                "batchsize_train": args.batch_size,
                "dimension": sum(p.numel() for p in model.parameters()),
                "percentile": args.percentile,
                "stride": args.stride,
                "bin_cnt": args.bin_cnt,
            })
        elif args.clipping == "dcsgde":
            make_private_kwargs.update({
                "batchsize_train": args.batch_size,
                "dimension": sum(p.numel() for p in model.parameters()),
                "percentile": args.percentile,
                "stride": args.stride,
                "bin_cnt": args.bin_cnt,
                "histogram_std": args.histogram_std,
            })

        model, optimizer, train_loader = privacy_engine.make_private(**make_private_kwargs)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, privacy_engine, epoch, stats)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_model.pt")

    if not args.disable_dp and len(stats["epsilon"]) > 0:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Clipping Threshold C", color="tab:red")
        ax1.plot(range(1, len(stats["clip_norm"]) + 1), stats["clip_norm"], color="tab:red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Epsilon", color="tab:blue")
        ax2.plot(range(1, len(stats["epsilon"]) + 1), stats["epsilon"], color="tab:blue", linestyle="--")
        fig.tight_layout()
        plt.title("Clipping Threshold and Epsilon Over Epochs")
        plt.grid(True)
        plt.savefig("cifar10_clip_and_epsilon.png")
        plt.show()


if __name__ == "__main__":
    main()
