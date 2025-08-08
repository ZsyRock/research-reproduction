#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from opacus.utils.epsilon_tracker import DynamicEpsilonTracker


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


def train(args, model, device, train_loader, optimizer, privacy_engine, epsilon_tracker, epoch, stats):

    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon_opacus = privacy_engine.accountant.get_epsilon(delta=args.delta)
        max_grad_norm = getattr(optimizer, "max_grad_norm", None)

        # 获取动态 C 下真实 epsilon
        if max_grad_norm:
            epsilon_tracker.step(current_C=max_grad_norm)
            epsilon_dynamic = epsilon_tracker.get_epsilon()
        else:
            epsilon_dynamic = None

        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε_opacus = {epsilon_opacus:.2f}, ε_dynamic = {epsilon_dynamic:.5f}, δ = {args.delta}) "
            f"Clip Norm: {max_grad_norm:.4f}" if max_grad_norm else ""
        )

        stats["epsilon"].append(epsilon_opacus)
        stats["epsilon_dynamic"].append(epsilon_dynamic)
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
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    # Training settings

    parser = argparse.ArgumentParser(
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--clipping",
        type=str,
        default="flat",
        choices=["flat", "per_layer", "adaptive", "dcsgdp", "dcsgde"],
        help="Type of clipping to use. 'dcsgdp' uses DCSGDPOptimizer and 'dcsgde' uses DCSGDEOptimizer",
    )
    parser.add_argument("--percentile", type=float, default=0.3, help="Percentile for clipping threshold estimation in DCSGDE")
    parser.add_argument("--stride", type=float, default=1.0, help="Stride of histogram bins")
    parser.add_argument("--bin-cnt", type=int, default=20, help="Number of bins in histogram")
    parser.add_argument("--histogram-std", type=float, default=6.0, help="Histogram noise std")

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = []

    stats = {
        "clip_norm": [],
        "epsilon": [],
        "epsilon_dynamic": [],
    }

    for _ in range(args.n_runs):
        model = SampleConvNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
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

            # 如果选择的是 DCSGDP，则添加特有参数
            if args.clipping == "dcsgdp":
                make_private_kwargs.update({
                    "batchsize_train": args.batch_size,
                    "dimension": sum(p.numel() for p in model.parameters()),
                })
            if args.clipping == "dcsgde":
                make_private_kwargs.update({
                    "batchsize_train": args.batch_size,
                    "dimension": sum(p.numel() for p in model.parameters()),
                    "percentile": args.percentile,
                    "stride": args.stride,
                    "bin_cnt": args.bin_cnt,
                    "histogram_std": args.histogram_std,
                })

            model, optimizer, train_loader = privacy_engine.make_private(**make_private_kwargs)
            # 初始化 epsilon tracker（动态 C 下真实 ε）
            epsilon_tracker = DynamicEpsilonTracker(
                delta=args.delta,
                sample_rate=args.batch_size / len(train_loader.dataset),
                noise_multiplier=args.sigma,
            )



        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine, epsilon_tracker, epoch, stats)
        run_results.append(test(model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"mnist_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")

    if not args.disable_dp and len(stats["epsilon"]) > 0:
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Clipping Threshold C", color="tab:red")
        ax1.plot(range(1, len(stats["clip_norm"]) + 1), stats["clip_norm"], color="tab:red", label="Clipping Threshold C")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Epsilon", color="tab:blue")
        ax2.plot(range(1, len(stats["epsilon"]) + 1), stats["epsilon"], color="tab:blue", linestyle="--", label="Epsilon")
        ax2.plot(range(1, len(stats["epsilon_dynamic"]) + 1),
                stats["epsilon_dynamic"], color="tab:green", linestyle=":", label="Epsilon (Dynamic C)")
        ax2.legend(loc="upper left")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        plt.title("Clipping Threshold and Epsilon Over Epochs (Mnist)")

        # 为顶部标题预留空间
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        plt.grid(True)
        plt.savefig("clip_and_epsilon_mnist.png")
        plt.show()
if __name__ == "__main__":
    main()
