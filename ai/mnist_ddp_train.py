import argparse
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SyntheticMnist(Dataset):
    """Deterministic MNIST-shaped fallback used only when explicitly enabled."""

    def __init__(self, size, seed):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(size, 1, 28, 28, generator=generator)
        self.labels = torch.randint(0, 10, (size,), generator=generator)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST CNN training with PyTorch DDP on Kubernetes")
    parser.add_argument("--mode", choices=["single", "ddp"], default="single")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-dir", default="/tmp/mnist")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rank_from_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    if "JOB_COMPLETION_INDEX" in os.environ:
        return int(os.environ["JOB_COMPLETION_INDEX"])
    return 0


def setup_distributed(args):
    if args.mode != "ddp":
        return 0, 1

    os.environ.setdefault("MASTER_ADDR", "mnist-ddp-master")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "2")
    os.environ.setdefault("RANK", str(get_rank_from_env()))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
    )
    return rank, world_size


def cleanup_distributed(args):
    if args.mode == "ddp" and dist.is_initialized():
        dist.destroy_process_group()


def build_datasets(args, rank):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    try:
        train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        dataset_source = "torchvision-mnist"
    except Exception as exc:
        if not args.allow_fallback:
            raise
        if rank == 0:
            print(f"WARNING: MNIST download failed, using synthetic fallback. reason={exc}", flush=True)
        train_size = args.limit_train if args.limit_train > 0 else 60000
        test_size = args.limit_test if args.limit_test > 0 else 10000
        train_set = SyntheticMnist(train_size, args.seed)
        test_set = SyntheticMnist(test_size, args.seed + 1)
        dataset_source = "synthetic-fallback"

    if args.limit_train > 0:
        train_set = Subset(train_set, range(min(args.limit_train, len(train_set))))
    if args.limit_test > 0:
        test_set = Subset(test_set, range(min(args.limit_test, len(test_set))))
    return train_set, test_set, dataset_source


def make_loaders(args, train_set, test_set, rank, world_size):
    if args.mode == "ddp":
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, test_loader, train_sampler


def train_one_epoch(model, device, loader, optimizer, epoch, sampler):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss = 0.0
    total_seen = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size
    return total_loss / max(total_seen, 1)


@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)

    rank, world_size = setup_distributed(args)
    device = torch.device("cpu")

    train_set, test_set, dataset_source = build_datasets(args, rank)
    train_loader, test_loader, sampler = make_loaders(args, train_set, test_set, rank, world_size)

    model = MnistCNN().to(device)
    if args.mode == "ddp":
        model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.mode == "ddp":
        dist.barrier()

    start = time.perf_counter()
    last_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        last_loss = train_one_epoch(model, device, train_loader, optimizer, epoch, sampler)
        print(
            f"epoch={epoch} mode={args.mode} rank={rank} world_size={world_size} "
            f"loss={last_loss:.4f}",
            flush=True,
        )

    if args.mode == "ddp":
        dist.barrier()
    train_seconds = time.perf_counter() - start

    accuracy = None
    if rank == 0:
        eval_model = model.module if isinstance(model, DDP) else model
        accuracy = evaluate(eval_model, device, test_loader)
        print(
            f"RESULT mode={args.mode} dataset={dataset_source} rank={rank} "
            f"world_size={world_size} epochs={args.epochs} "
            f"train_samples={len(train_set)} test_samples={len(test_set)} "
            f"per_worker_batch_size={args.batch_size} "
            f"train_seconds={train_seconds:.3f} "
            f"accuracy={accuracy:.4f} final_loss={last_loss:.4f}",
            flush=True,
        )
    else:
        print(
            f"RESULT mode={args.mode} rank={rank} world_size={world_size} "
            f"epochs={args.epochs} train_seconds={train_seconds:.3f} final_loss={last_loss:.4f}",
            flush=True,
        )

    cleanup_distributed(args)


if __name__ == "__main__":
    main()
