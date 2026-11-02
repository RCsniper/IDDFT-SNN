import os
import gc
import csv
import json
import time
import argparse
import random
from warnings import warn
from contextlib import contextmanager

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate


# ============================================================
# Reproducibility
# ============================================================

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Cutout
# ============================================================

class Cutout:
    def __init__(self, length=16, fill=0.0):
        self.length = length
        self.fill = fill

    def __call__(self, img):
        c, h, w = img.shape
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        half = self.length // 2
        y1 = max(0, y - half)
        y2 = min(h, y + half)
        x1 = max(0, x - half)
        x2 = min(w, x + half)

        if isinstance(self.fill, torch.Tensor):
            img[:, y1:y2, x1:x2] = self.fill.to(img.device, img.dtype)
        else:
            img[:, y1:y2, x1:x2] = float(self.fill)

        return img


# ============================================================
# IDDFT Gaussian bfunc
# Same as your real training code:
# bfunc = lambda t: torch.exp(-(amp * (t + phase) ** 2))
# ============================================================

def make_gaussian_bfunc(amp=3.0, phase=-1.5):
    def bfunc(t):
        return torch.exp(-(amp * (t + phase) ** 2))
    return bfunc


# ============================================================
# IDDFT spike function
# ============================================================

class Augment_spike_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem_thr, v, bfunc):
        spk = (mem_thr > 0).to(mem_thr.dtype)
        ctx.bfunc = bfunc
        ctx.save_for_backward(v)
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        g = ctx.bfunc(v)
        grad_v = grad_output * g

        # No gradient to membrane-potential path.
        # Gradient only goes to current input drive v.
        return None, grad_v, None


class Leaky_augment(nn.Module):

    reset_dict = {"subtract": 0, "zero": 1, "none": 2}

    def __init__(
        self,
        beta,
        bfunc,
        threshold=1.0,
        reset_mechanism="subtract",
        reset_delay=True,
        graded_spikes_factor=1.0,
        inhibition=False,
        output=False,
        init_hidden=False,
        detach_state=True,
    ):
        super().__init__()

        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("threshold", torch.as_tensor(threshold))
        self.register_buffer(
            "graded_spikes_factor",
            torch.as_tensor(graded_spikes_factor),
        )

        self.bfunc = bfunc
        self.inhibition = inhibition
        self.output = output
        self.init_hidden = init_hidden
        self.reset_delay = reset_delay
        self.detach_state = detach_state

        if reset_mechanism not in self.reset_dict:
            raise ValueError("reset_mechanism must be 'subtract', 'zero', or 'none'")

        self.register_buffer(
            "reset_mechanism_val",
            torch.as_tensor(self.reset_dict[reset_mechanism]),
        )
        self._reset_mechanism = reset_mechanism

        self.register_buffer("mem", torch.zeros(0), persistent=False)

        if inhibition:
            warn("Inhibition is not well tested for convolutional layers.", UserWarning)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    def init_leaky(self):
        return self.reset_mem()

    def forward(self, input_, mem=None):
        if mem is None:
            if self.mem.shape != input_.shape:
                mem = torch.zeros_like(input_)
            else:
                mem = self.mem
        else:
            if mem.shape != input_.shape:
                mem = torch.zeros_like(input_)

        prev_mem = mem.detach() if self.detach_state else mem

        # IDDFT does not need membrane potential in backward.
        # Therefore, membrane dynamics are computed without graph construction.
        with torch.no_grad():
            reset = ((prev_mem - self.threshold) > 0).to(input_.dtype)
            beta = self.beta.clamp(0, 1)

            if self.reset_mechanism_val == 0:  # subtract
                new_mem = beta * prev_mem + input_.detach() - reset * self.threshold
            elif self.reset_mechanism_val == 1:  # zero
                new_mem = beta * ((1 - reset) * prev_mem) + input_.detach()
            else:  # none
                new_mem = beta * prev_mem + input_.detach()

            mem_shift = new_mem - self.threshold

        spk = Augment_spike_function.apply(mem_shift, input_, self.bfunc)
        spk = spk * self.graded_spikes_factor

        if not self.reset_delay:
            with torch.no_grad():
                do_reset = (spk.detach() / self.graded_spikes_factor - reset)

                if self.reset_mechanism_val == 0:
                    new_mem = new_mem - do_reset * self.threshold
                elif self.reset_mechanism_val == 1:
                    new_mem = new_mem - do_reset * new_mem

        mem_out = new_mem.detach() if self.detach_state else new_mem
        self.mem = mem_out

        if self.output:
            return spk, mem_out
        elif self.init_hidden:
            return spk
        else:
            return spk, mem_out


def lif_forward(lif, x, mem):
    if mem is None:
        mem = torch.zeros_like(x)
    elif mem.shape != x.shape:
        mem = torch.zeros_like(x)

    out = lif(x, mem)

    if isinstance(out, tuple):
        spk, new_mem = out
    else:
        spk, new_mem = out, None

    return spk, new_mem


@contextmanager
def preserve_bn_running_stats(model):
    snapshots = []

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            item = {
                "module": m,
                "running_mean": None if m.running_mean is None else m.running_mean.detach().clone(),
                "running_var": None if m.running_var is None else m.running_var.detach().clone(),
                "num_batches_tracked": None,
            }

            if hasattr(m, "num_batches_tracked") and m.num_batches_tracked is not None:
                item["num_batches_tracked"] = m.num_batches_tracked.detach().clone()

            snapshots.append(item)

    try:
        yield
    finally:
        for item in snapshots:
            m = item["module"]

            if item["running_mean"] is not None:
                m.running_mean.copy_(item["running_mean"])

            if item["running_var"] is not None:
                m.running_var.copy_(item["running_var"])

            if item["num_batches_tracked"] is not None:
                m.num_batches_tracked.copy_(item["num_batches_tracked"])


# ============================================================
# No-tdBN ResNet19 SNN
# ============================================================

class BasicBlockSNN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, lif_factory):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = lif_factory()

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = lif_factory()

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward_step(self, x, mem1, mem2):
        identity = x

        out = self.bn1(self.conv1(x))
        out, mem1 = lif_forward(self.lif1, out, mem1)

        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out, mem2 = lif_forward(self.lif2, out, mem2)

        return out, mem1, mem2


class ResNet19_SNN(nn.Module):
    """
    BPTT+SG:
        method="bptt_sg"
        uses snn.Leaky, no detach, full temporal graph.

    IDDFT+uSF:
        method="bp_iddft"
        uses Leaky_augment, detaches historical membrane states.
    """

    def __init__(
        self,
        num_classes=10,
        method="bptt_sg",
        beta=0.5,
        threshold=1.0,
        bfunc=None,
        spike_grad=None,
        num_steps=4,
    ):
        super().__init__()

        if method == "bp_iddft":
            if bfunc is None:
                raise ValueError("bfunc must be provided for IDDFT+uSF.")

            def lif_factory():
                return Leaky_augment(
                    beta=beta,
                    threshold=threshold,
                    bfunc=bfunc,
                    detach_state=True,
                )

        elif method == "bptt_sg":
            def lif_factory():
                return snn.Leaky(
                    beta=beta,
                    threshold=threshold,
                    spike_grad=spike_grad,
                )

        else:
            raise ValueError("method must be 'bptt_sg' or 'bp_iddft'.")

        self.method = method
        self.num_steps = num_steps
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.lif0 = lif_factory()

        self.layer1 = self._make_layer(planes=128, num_blocks=3, stride=1, lif_factory=lif_factory)
        self.layer2 = self._make_layer(planes=256, num_blocks=3, stride=2, lif_factory=lif_factory)
        self.layer3 = self._make_layer(planes=512, num_blocks=2, stride=2, lif_factory=lif_factory)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlockSNN.expansion, num_classes)
        self.lif_out = lif_factory()

        self._initialize_weights()

    def _make_layer(self, planes, num_blocks, stride, lif_factory):
        blocks = nn.ModuleList()
        strides = [stride] + [1] * (num_blocks - 1)

        for st in strides:
            blocks.append(
                BasicBlockSNN(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=st,
                    lif_factory=lif_factory,
                )
            )
            self.in_planes = planes * BasicBlockSNN.expansion

        return blocks

    def init_states(self):
        return {
            "mem0": None,
            "mem_out": None,
            "l1_1": [None] * len(self.layer1),
            "l1_2": [None] * len(self.layer1),
            "l2_1": [None] * len(self.layer2),
            "l2_2": [None] * len(self.layer2),
            "l3_1": [None] * len(self.layer3),
            "l3_2": [None] * len(self.layer3),
        }

    def forward_one_step(self, x, states):
        out = self.bn1(self.conv1(x))
        out, states["mem0"] = lif_forward(self.lif0, out, states["mem0"])

        for i, blk in enumerate(self.layer1):
            out, states["l1_1"][i], states["l1_2"][i] = blk.forward_step(
                out,
                states["l1_1"][i],
                states["l1_2"][i],
            )

        for i, blk in enumerate(self.layer2):
            out, states["l2_1"][i], states["l2_2"][i] = blk.forward_step(
                out,
                states["l2_1"][i],
                states["l2_2"][i],
            )

        for i, blk in enumerate(self.layer3):
            out, states["l3_1"][i], states["l3_2"][i] = blk.forward_step(
                out,
                states["l3_1"][i],
                states["l3_2"][i],
            )

        out = self.ap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        spk, states["mem_out"] = lif_forward(self.lif_out, out, states["mem_out"])

        return spk, states

    def forward_count(self, x):
        """
        Return spike count logits: sum_t spk_t.
        This matches ce_count_loss-style training more closely than mean logits.
        """
        states = self.init_states()
        count_logits = None

        for _ in range(self.num_steps):
            spk, states = self.forward_one_step(x, states)
            count_logits = spk if count_logits is None else count_logits + spk

        return count_logits

    @torch.no_grad()
    def forward_count_nograd(self, x):
        return self.forward_count(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ============================================================
# Dataset
# ============================================================

def build_dataset(dataset_name, cifar10_root, cifar100_root):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        num_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        dataset_cls = datasets.CIFAR10
        data_root = cifar10_root

    elif dataset_name == "cifar100":
        num_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        dataset_cls = datasets.CIFAR100
        data_root = cifar100_root

    else:
        raise ValueError("dataset_name must be cifar10 or cifar100")

    mean_t = torch.tensor(mean).view(3, 1, 1)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        Cutout(length=16, fill=mean_t),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = dataset_cls(
        data_root,
        train=True,
        download=False,
        transform=train_transform,
    )

    test_set = dataset_cls(
        data_root,
        train=False,
        download=False,
        transform=test_transform,
    )

    return train_set, test_set, num_classes


# ============================================================
# Optimizer
# ============================================================

def make_optimizer(net, lr, weight_decay):
    decay, no_decay = [], []

    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue

        if (
            p.ndim == 1
            or name.endswith(".bias")
            or "bn" in name.lower()
            or "tdBN" in name
        ):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = optim.SGD(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=0.9,
        nesterov=True,
    )

    return optimizer


# ============================================================
# Train / Eval
# ============================================================

def train_batch_bptt(net, images, targets, optimizer, criterion):
    optimizer.zero_grad(set_to_none=True)

    count_logits = net.forward_count(images)
    loss = criterion(count_logits, targets)

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = count_logits.argmax(dim=1)
        correct = pred.eq(targets).sum().item()

    return loss.item(), correct, targets.size(0)


def train_batch_iddft_twopass(net, images, targets, optimizer, criterion):

    optimizer.zero_grad(set_to_none=True)


    with preserve_bn_running_stats(net):
        with torch.no_grad():
            count_logits = net.forward_count_nograd(images)

    count_leaf = count_logits.detach().requires_grad_(True)
    loss = criterion(count_leaf, targets)
    loss.backward()

    grad_count = count_leaf.grad.detach()

    with torch.no_grad():
        pred = count_logits.argmax(dim=1)
        correct = pred.eq(targets).sum().item()


    states = net.init_states()

    for _ in range(net.num_steps):
        spk, states = net.forward_one_step(images, states)

        torch.autograd.backward(
            tensors=spk,
            grad_tensors=grad_count,
            retain_graph=False,
            create_graph=False,
        )

        del spk

    optimizer.step()

    return loss.item(), correct, targets.size(0)


@torch.no_grad()
def evaluate(net, test_loader, criterion, device):
    net.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        count_logits = net.forward_count_nograd(images)
        loss = criterion(count_logits, targets)

        pred = count_logits.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)
        loss_sum += loss.item() * targets.size(0)

    return loss_sum / total, 100.0 * correct / total


def train_one_run(args, dataset_name, method):
    device = torch.device(args.device)

    train_set, test_set, num_classes = build_dataset(
        dataset_name,
        args.cifar10_root,
        args.cifar100_root,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    if method == "bptt_sg":
        spike_grad = surrogate.atan(alpha=2.0)
        bfunc = None
    elif method == "bp_iddft":
        spike_grad = None
        bfunc = make_gaussian_bfunc(amp=args.amp, phase=args.phase)
    else:
        raise ValueError("method must be bptt_sg or bp_iddft")

    net = ResNet19_SNN(
        num_classes=num_classes,
        method=method,
        beta=args.beta,
        threshold=args.threshold,
        bfunc=bfunc,
        spike_grad=spike_grad,
        num_steps=args.T_step,
    ).to(device)

    optimizer = make_optimizer(net, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.n_epochs,
        eta_min=args.eta_min,
    )

    criterion = nn.CrossEntropyLoss()

    save_dir = os.path.join(
        args.save_root,
        dataset_name,
        f"T{args.T_step}",
        method,
        f"seed_{args.seed}",
    )
    os.makedirs(save_dir, exist_ok=True)

    config = vars(args).copy()
    config["dataset"] = dataset_name
    config["method"] = method
    config["num_classes"] = num_classes

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(config, f, indent=4)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
        "epoch_time_sec": [],
        "peak_memory_GB": [],
    }

    best_acc = 0.0
    best_epoch = -1

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    total_start = time.perf_counter()

    print("=" * 100)
    print(f"Dataset    : {dataset_name}")
    print(f"Method     : {method}")
    print(f"Epochs     : {args.n_epochs}")
    print(f"Batch      : {args.batch}")
    print(f"T_step     : {args.T_step}")
    print(f"LR         : {args.lr}")
    print(f"WD         : {args.weight_decay}")
    print(f"beta       : {args.beta}")
    print(f"threshold  : {args.threshold}")
    if method == "bp_iddft":
        print(f"Gaussian   : exp(-(amp * (x + phase)^2)), amp={args.amp}, phase={args.phase}")
        print("Backward   : efficient two-pass step-wise backward")
    else:
        print("Backward   : full BPTT+SG")
    print("=" * 100, flush=True)

    for epoch in range(args.n_epochs):
        net.train()

        epoch_start = time.perf_counter()

        train_total = 0
        train_correct = 0
        train_loss_sum = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if method == "bptt_sg":
                loss_value, correct, n = train_batch_bptt(
                    net,
                    images,
                    targets,
                    optimizer,
                    criterion,
                )
            else:
                loss_value, correct, n = train_batch_iddft_twopass(
                    net,
                    images,
                    targets,
                    optimizer,
                    criterion,
                )

            train_loss_sum += loss_value * n
            train_correct += correct
            train_total += n

            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                print(
                    f"Epoch [{epoch+1:03d}/{args.n_epochs}] "
                    f"Batch [{batch_idx:04d}/{len(train_loader)}] "
                    f"loss={loss_value:.4f} "
                    f"peak_memory={peak_memory:.2f}G",
                    flush=True,
                )

        scheduler.step()

        torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start

        train_loss = train_loss_sum / train_total
        train_acc = 100.0 * train_correct / train_total

        test_loss, test_acc = evaluate(net, test_loader, criterion, device)

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        lr_now = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr_now)
        history["epoch_time_sec"].append(epoch_time)
        history["peak_memory_GB"].append(peak_memory)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save(net.state_dict(), os.path.join(save_dir, "best_model.pth"))

        print(
            f"[Epoch {epoch+1:03d}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.2f}% "
            f"best={best_acc:.2f}%@{best_epoch} "
            f"time={epoch_time:.1f}s "
            f"peak_memory={peak_memory:.2f}G",
            flush=True,
        )

        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

    torch.cuda.synchronize(device)
    total_time = time.perf_counter() - total_start
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 3

    summary = {
        "dataset": dataset_name,
        "method": method,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_acc": history["test_acc"][-1],
        "total_time_sec": total_time,
        "total_time_hour": total_time / 3600.0,
        "peak_memory_GB": peak_memory,
    }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    torch.save(net.state_dict(), os.path.join(save_dir, "last_model.pth"))

    print("=" * 100)
    print(
        f"[DONE] dataset={dataset_name} method={method} "
        f"best_acc={best_acc:.2f}%@{best_epoch} "
        f"final_acc={history['test_acc'][-1]:.2f}% "
        f"time={total_time/3600.0:.2f}h "
        f"peak_memory={peak_memory:.2f}G"
    )
    print("=" * 100, flush=True)

    del net
    del optimizer
    del train_loader
    del test_loader
    del train_set
    del test_set

    gc.collect()
    torch.cuda.empty_cache()

    return summary


def save_global_summary(summaries, save_root):
    os.makedirs(save_root, exist_ok=True)
    path = os.path.join(save_root, "global_summary.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "best_acc",
                "best_epoch",
                "final_acc",
                "total_time_sec",
                "total_time_hour",
                "peak_memory_GB",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Saved global summary to: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cifar10_root",
        type=str,
        default="...",
    )
    parser.add_argument(
        "--cifar100_root",
        type=str,
        default="...",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10"],
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bptt_sg", "bp_iddft"],
        choices=["bptt_sg", "bp_iddft"],
    )

    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=5e-5)

    parser.add_argument("--T_step", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=1.0)

    parser.add_argument("--amp", type=float, default=3.0)
    parser.add_argument("--phase", type=float, default=-1.5)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument(
        "--save_root",
        type=str,
        default="...",
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
        torch.cuda.set_device(device.index if device.index is not None else 0)
    else:
        raise RuntimeError("CUDA is required for this training script.")

    set_global_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    summaries = []

    for dataset_name in args.datasets:
        for method in args.methods:
            summary = train_one_run(args, dataset_name, method)
            summaries.append(summary)

    save_global_summary(summaries, args.save_root)


if __name__ == "__main__":
    main()