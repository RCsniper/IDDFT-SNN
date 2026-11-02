import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import math
import random
import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

# from biotorch.layers.usf import Conv2d, Linear

from ResNet_19 import ResNet19


parser = argparse.ArgumentParser()
#parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--trial_num', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_epochs', type=int, default=400)
parser.add_argument('--batch', type=int, default=128)

parser.add_argument('--training_strategy', type=str, default='auSF') #BP, uSF, auSF
parser.add_argument('--bfunc', type=str, default='gaussian')
parser.add_argument('--amp', type=float, default=3.0)
parser.add_argument('--phase', type=float, default=-1.5)
parser.add_argument('--device', type=str, default='cuda:4')

parser.add_argument('--T_step', type=int, default=4)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--threshold', type=float, default=1.0)

args = parser.parse_args()

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
#
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(args.trial_num * 100 + args.trial_num)
log_root = "..."
#
# if args.training_strategy == 'BP':
#     save_root = "..."
# elif args.training_strategy == 'uSF':
#     save_root = "..."
# elif args.training_strategy == 'auSF':
#     save_root = "..."
# os.makedirs(save_root, exist_ok=True)

data_path=log_root+'/data/cifar_100'
dtype = torch.float
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device.index)
# Temporal Dynamics
num_steps = args.T_step
beta = args.beta
threshold = args.threshold

bfunc=None
if args.training_strategy == 'auSF':
    bfunc = lambda t: torch.exp(-(args.amp * (t + args.phase) ** 2))

# spike_grad = surrogate.fast_sigmoid(slope=25)
spike_grad = surrogate.atan(alpha=2.0)

#
if args.training_strategy == 'auSF':
    net = ResNet19(num_classes=100, training_strategy=args.training_strategy, beta=args.beta,
                       threshold=args.threshold, bfunc=bfunc, T=args.T_step
                       ).to(device)
else:
    net = ResNet19(
        T=args.T_step, num_classes=100,
        training_strategy=args.training_strategy,
        beta=args.beta, threshold=args.threshold,
        spike_grad=spike_grad,
    ).to(device)

print(net)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

decay, no_decay = [], []
for name, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 1 or name.endswith(".bias") or "tdBN" in name or "bn" in name.lower():
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.SGD(
    [{"params": decay, "weight_decay": 1e-4},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, momentum=0.9, nesterov=True
)

@torch.no_grad()
def bn_calibrate(model, loader, device, num_batches=200):
    model.train()

    n = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)  # 只 forward
        n += 1
        if n >= num_batches:
            break
    model.eval()

# args_={
#     "trial_num": args.trial_num,
#     "epochs": args.n_epochs,
#     "batch_size": args.batch,
#     "lr": args.lr,
#     "weight_decay": args.weight_decay,
#     "training_strategy": args.training_strategy,
#     "T_step": args.T_step,
#     "beta": args.beta,
#     "threshold": args.threshold,
#     "model": repr(net),
#     "conv_type": str(type(net.encode[0].layer)),
#     "linear_type": str(type(net.fc[0].layer)),
# }
#
# if args.training_strategy == 'auSF':
#     args_['bfunc'] = args.bfunc
#     args_['amp'] = args.amp
#     args_['phase'] = args.phase
# else:
#     args_['bfunc'] = str(spike_grad)
#
#
# with open(save_root+'/args.json', 'w') as json_file:
#     json.dump(args_, json_file, indent=4)


mean = (0.5071, 0.4867, 0.4408)
std  = (0.2675, 0.2565, 0.2761)

mean_t = torch.tensor(mean).view(3,1,1)


class Cutout:
    """
    img: Tensor [C,H,W] after ToTensor()
    length: int or (min_len, max_len)
    p: apply probability
    fill: float or Tensor [C,1,1] (e.g., channel mean in *unnormalized* space)
    """
    def __init__(self, length=(8, 12), p=0.5, fill=0.0):
        self.length = length
        self.p = p
        self.fill = fill

    def __call__(self, img):
        if random.random() > self.p:
            return img

        c, h, w = img.shape

        if isinstance(self.length, (tuple, list)):
            L = random.randint(int(self.length[0]), int(self.length[1]))
        else:
            L = int(self.length)

        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        half = L // 2
        y1 = max(0, y - half)
        y2 = min(h, y + half)
        x1 = max(0, x - half)
        x2 = min(w, x + half)


        img = img.clone()

        if isinstance(self.fill, torch.Tensor):
            img[:, y1:y2, x1:x2] = self.fill.to(img.device, img.dtype)
        else:
            img[:, y1:y2, x1:x2] = float(self.fill)
        return img


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    Cutout(length=(16, 16), p=0.75, fill=mean_t),
    transforms.Normalize(mean, std),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

cifar100_train = datasets.CIFAR100(data_path, train=True, download=True, transform=train_transform)
cifar100_test = datasets.CIFAR100(data_path, train=False, download=True, transform=test_transform)

train_loader = DataLoader(cifar100_train, batch_size=args.batch,
                          shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4,)
test_loader = DataLoader(cifar100_test, batch_size=args.batch,
                         shuffle=False, drop_last=False,
                         num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4,)



num_epochs = args.n_epochs
loss_hist = []
test_loss_hist = []
train_acc_hist = []
test_acc_hist = []

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.5, patience=3, threshold=1e-4, min_lr=5e-5
# )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)

test_best = 0.0
for epoch in range(num_epochs):
    # ---------------- train ----------------
    net.train()
    train_total = 0
    train_correct = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # spk_rec = net(data)
        # logits = net.forward_logits(spk_rec, reduce="mean")
        logits = net.forward_logits(data, reduce="mean")
        loss_val = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

        if batch_idx % 100 == 0:
            print(f"Step [{batch_idx}/{len(train_loader)}], loss: {loss_val.item()}")

        with torch.no_grad():

            pred = logits.argmax(dim=1)  # [B]
            train_correct += (pred == targets).sum().item()
            train_total += targets.size(0)
    train_acc_hist.append(train_correct / train_total)

    # ---------------- test ----------------
    bn_calibrate(net, train_loader, device, num_batches=200)
    net.eval()
    with torch.no_grad():
        total = 0
        acc = 0
        test_loss_sum = 0.0

        for data, targets in test_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits_T = net(data)  # [B,T,C]
            logits = logits_T.mean(dim=1)
            pred = logits.argmax(dim=1)
            loss_batch = loss_fn(logits, targets)

            acc += (pred == targets).sum().item()
            total += targets.size(0)
            test_loss_sum += loss_batch.item() * targets.size(0)

        loss_test = test_loss_sum / total

    print(f"Epoch: {epoch}, Train Accuracy: {100*train_correct / train_total:.2f}% Test Accuracy: {100 * acc / total:.2f}%")
    test_acc = acc / total
    test_acc_hist.append(test_acc)

    if test_acc > test_best:
        test_best = test_acc
        # torch.save(net.state_dict(), save_root + '/model.pth')

    # scheduler.step(loss_test)
    scheduler.step()
print(max(test_acc_hist))

# data={
#     "best acc": max(test_acc_hist),
#     "train loss": loss_hist,
#     "train acc": train_acc_hist,
#     "test acc": test_acc_hist,
# }
# with open(save_root+'/data.json', 'w') as json_file:
#     json.dump(data, json_file, indent=4)


