import os
import sys
import time
import random
import argparse
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import snntorch as snn
from snntorch import surrogate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ResNet_19 import ResNet19


# -------------------------
# perf flags
# -------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# -------------------------
# args
# -------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--trial_num', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=32)  # per-GPU batch under torchrun

parser.add_argument('--training_strategy', type=str, default='auSF')  # BP, uSF, auSF
parser.add_argument('--bfunc', type=str, default='gaussian')
parser.add_argument('--amp', type=float, default=3.0)
parser.add_argument('--phase', type=float, default=-1.5)
parser.add_argument('--device', type=str, default='cuda:0')  # ignored under torchrun
parser.add_argument('--T_step', type=int, default=4)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--threshold', type=float, default=1.0)

parser.add_argument(
    '--data_path',
    type=str,
    default='...'
)
parser.add_argument('--num_classes', type=int, default=200)
parser.add_argument('--img_size', type=int, default=64)

parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--log_interval', type=int, default=1)

args = parser.parse_args()


# -------------------------
# distributed init
# -------------------------
def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device.index)
        distributed = False

    return device, rank, world_size, local_rank, distributed


device, rank, world_size, local_rank, distributed = init_distributed()


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(args.trial_num * 100 + args.trial_num)


class TinyImageNetValDataset(Dataset):
    def __init__(self, val_root, class_to_idx, transform=None):
        self.transform = transform
        self.samples = []

        ann_path = os.path.join(val_root, "val_annotations.txt")
        img_dir = os.path.join(val_root, "images")

        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Cannot find {ann_path}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Cannot find {img_dir}")

        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_name = parts[0]
                wnid = parts[1]

                if wnid not in class_to_idx:
                    continue

                img_path = os.path.join(img_dir, img_name)
                target = class_to_idx[wnid]
                self.samples.append((img_path, target))

        if len(self.samples) == 0:
            raise RuntimeError("No validation samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target



bfunc = None
if args.training_strategy == 'auSF':
    bfunc = lambda t: torch.exp(-(args.amp * (t + args.phase) ** 2))

spike_grad = surrogate.atan(alpha=2.0)

if args.training_strategy == 'auSF':
    model = ResNet19(
        num_classes=args.num_classes,
        training_strategy=args.training_strategy,
        beta=args.beta,
        threshold=args.threshold,
        bfunc=bfunc,
        T=args.T_step,
    )
else:
    model = ResNet19(
        T=args.T_step,
        num_classes=args.num_classes,
        training_strategy=args.training_strategy,
        beta=args.beta,
        threshold=args.threshold,
        spike_grad=spike_grad,
    )

model = model.to(device)

if rank == 0:
    print("Model created.", flush=True)
    print("Training mode: torchrun multi-GPU, manual gradient synchronization", flush=True)
    print("Dataset: Tiny-ImageNet-200", flush=True)
    print(f"world_size = {world_size}", flush=True)
    print(f"num_classes = {args.num_classes}", flush=True)
    print(f"image size = {args.img_size}x{args.img_size}", flush=True)
    print(f"T_step = {args.T_step}", flush=True)
    print(type(model.encode[0].layer), flush=True)
    print(type(model.fc[0].layer), flush=True)
    print(model.encode[0].layer.__class__.__module__, flush=True)
    print(model.fc[0].layer.__class__.__module__, flush=True)


# -------------------------
# loss / optimizer / scheduler
# -------------------------
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 1 or name.endswith(".bias") or "tdBN" in name or "bn" in name.lower():
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.SGD(
    [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=args.lr,
    momentum=0.9,
    nesterov=True,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.n_epochs,
    eta_min=1e-6,
)


use_amp = False
scaler = None


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

traindir = os.path.join(args.data_path, "train")
valdir = os.path.join(args.data_path, "val")

train_transform = transforms.Compose([
    transforms.RandomCrop(args.img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

trainset = datasets.ImageFolder(root=traindir, transform=train_transform)

testset = datasets.ImageFolder(root=valdir, transform=test_transform)

if rank == 0:
    print(f"Train samples: {len(trainset)}", flush=True)
    print(f"Val samples: {len(testset)}", flush=True)
    print(f"Train classes: {len(trainset.classes)}", flush=True)

train_sampler = DistributedSampler(
    trainset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
) if distributed else None

test_sampler = DistributedSampler(
    testset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
) if distributed else None

loader_kwargs = dict(
    num_workers=args.num_workers,
    pin_memory=True,
)

if args.num_workers > 0:
    loader_kwargs.update(
        persistent_workers=True,
        prefetch_factor=4,
    )

train_loader = DataLoader(
    trainset,
    batch_size=args.batch,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    drop_last=True,
    **loader_kwargs,
)

test_loader = DataLoader(
    testset,
    batch_size=args.batch,
    shuffle=False,
    sampler=test_sampler,
    drop_last=False,
    **loader_kwargs,
)


# -------------------------
# distributed helpers
# -------------------------
@torch.no_grad()
def reduce_int(x):
    t = torch.tensor([x], device=device, dtype=torch.long)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


@torch.no_grad()
def reduce_float(x):
    t = torch.tensor([x], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


@torch.no_grad()
def average_gradients(model):
    if not distributed:
        return

    # Synchronize every trainable parameter in the same order on every rank.
    for p in model.parameters():
        if not p.requires_grad:
            continue

        if p.grad is None:
            p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)

        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


# -------------------------
# training loop
# -------------------------
test_acc_lst = []
for epoch in range(args.n_epochs):
    if distributed:
        train_sampler.set_epoch(epoch)

    model.train()
    train_correct = 0
    train_total = 0

    if rank == 0:
        last_t = time.perf_counter()
        last_step = 0

    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data.to(device, non_blocking=True).contiguous()
        targets = targets.to(device, non_blocking=True)


        optimizer.zero_grad(set_to_none=False)


        logits_T = model(data)
        logits = logits_T.mean(dim=1)
        loss_val = loss_fn(logits, targets)


        loss_val.backward()

        if distributed:

            average_gradients(model)


        torch.cuda.synchronize()


        optimizer.step()


        torch.cuda.synchronize()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            train_correct += (pred == targets).sum().item()
            train_total += targets.size(0)

        if rank == 0 and (batch_idx + 1) % args.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()

            now = time.perf_counter()
            steps = (batch_idx + 1) - last_step
            dt = now - last_t
            imgs = steps * args.batch * world_size
            ips = imgs / max(dt, 1e-9)

            print(
                f"Epoch {epoch} Step [{batch_idx + 1}/{len(train_loader)}] "
                f"loss={loss_val.item():.4f} images/sec={ips:,.1f} "
                f"(global_batch={args.batch * world_size})",
                flush=True,
            )

            last_t = now
            last_step = batch_idx + 1

    train_correct_g = reduce_int(train_correct)
    train_total_g = reduce_int(train_total)
    train_acc = train_correct_g / max(1, train_total_g)

    model.eval()
    test_correct = 0
    test_total = 0
    test_loss_sum = 0.0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device, non_blocking=True).contiguous()
            targets = targets.to(device, non_blocking=True)

            logits_T = model(data)
            logits = logits_T.mean(dim=1)
            loss_batch = loss_fn(logits, targets)

            pred = logits.argmax(dim=1)
            test_correct += (pred == targets).sum().item()
            test_total += targets.size(0)
            test_loss_sum += loss_batch.item() * targets.size(0)

    test_correct_g = reduce_int(test_correct)
    test_total_g = reduce_int(test_total)
    test_loss_sum_g = reduce_float(test_loss_sum)

    test_loss = test_loss_sum_g / max(1, test_total_g)
    test_acc = test_correct_g / max(1, test_total_g)
    test_acc_lst.append(test_acc)
    if rank == 0:
        print(
            f"Epoch {epoch} DONE: "
            f"Train Acc={100 * train_acc:.2f}% "
            f"Val Acc={100 * test_acc:.2f}% "
            f"Val Loss={test_loss:.4f}",
            flush=True,
        )

    scheduler.step()
print(max(test_acc_lst))
if distributed:
    dist.destroy_process_group()