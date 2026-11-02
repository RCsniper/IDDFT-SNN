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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from snntorch import surrogate

from VGGSNN import VGGSNN
from torchvision.transforms import InterpolationMode


parser = argparse.ArgumentParser()
parser.add_argument('--trial_num', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_epochs', type=int, default=400)
parser.add_argument('--batch', type=int, default=50)

parser.add_argument('--training_strategy', type=str, default='auSF')  # BP, uSF, auSF
parser.add_argument('--bfunc', type=str, default='gaussian')
parser.add_argument('--amp', type=float, default=2.5)
parser.add_argument('--phase', type=float, default=-1.0)
parser.add_argument('--device', type=str, default='cuda:2')

parser.add_argument('--T_step', type=int, default=10)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--threshold', type=float, default=1.0)

args = parser.parse_args()


CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)
CLASS_TO_LABEL = {c: i for i, c in enumerate(CLASS_NAMES)}

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(args.trial_num * 100 + args.trial_num)


class MyCutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(-2)
        w = img.size(-1)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).to(img.device, img.dtype)
        mask = mask.expand_as(img)
        return img * mask


def function_nda(data, M=1, N=2):
    c = 15 * N
    rotate_tf = transforms.RandomRotation(
        degrees=c,
        interpolation=InterpolationMode.NEAREST
    )
    e = 8 * N
    cutout_tf = MyCutout(length=e)

    def roll(data, N=1):
        a = N * 2 + 1
        off1 = np.random.randint(-a, a + 1)
        off2 = np.random.randint(-a, a + 1)
        return torch.roll(data, shifts=(off1, off2), dims=(2, 3))

    def rotate(data, N):
        return rotate_tf(data)

    def cutout(data, N):
        return cutout_tf(data)

    transforms_list = [roll, rotate, cutout]
    sampled_ops = np.random.choice(transforms_list, M)
    for op in sampled_ops:
        data = op(data, N)
    return data




transform_train = transforms.Compose([
    #ToTensor(),
    transforms.Resize(size=(48, 48)),
    transforms.RandomCrop(48, padding=4),
    transforms.RandomHorizontalFlip()
])

transform_test = transforms.Compose([
    #ToTensor(),
    transforms.Resize(size=(48, 48))
])

class CIFAR10DVSNPZDataset(Dataset):
    def __init__(self, root, train=True, train_ratio=0.9, transform=None, seed=0):

        self.root = root
        self.transform = transform
        self.samples = []

        rng = random.Random(seed)

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            files = sorted([
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.endswith(".npz")
            ])

            rng.shuffle(files)
            split = math.ceil(len(files) * train_ratio)

            if train:
                selected = files[:split]
            else:
                selected = files[split:]

            label = CLASS_TO_LABEL[class_name]
            for f in selected:
                self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_name, label = self.samples[index]

        obj = np.load(file_name)
        data = obj["frames"]   # [T, 2, 128, 128]
        data = torch.from_numpy(data).float()

        if self.transform is not None:
            data = self.transform(data)

        return data, label

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


frames_root = "..."
num_classes = 10

# if args.training_strategy == 'BP':
#     save_root = "..."
# elif args.training_strategy == 'uSF':
#     save_root = "..."
# elif args.training_strategy == 'auSF':
#     save_root = "..."
# os.makedirs(save_root, exist_ok=True)

trainset = CIFAR10DVSNPZDataset(frames_root, train=True, transform=transform_train, seed=0)
testset = CIFAR10DVSNPZDataset(frames_root, train=False, transform=transform_test, seed=0)

g = torch.Generator()
# g.manual_seed(101)
g.manual_seed(args.trial_num * 100 + args.trial_num)

train_loader = DataLoader(
    trainset,
    batch_size=args.batch,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=True,
    prefetch_factor=4,
)

test_loader = DataLoader(
    testset,
    batch_size=args.batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    worker_init_fn=seed_worker,
    persistent_workers=True,
    prefetch_factor=4,
)


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
    net = VGGSNN(
        T=args.T_step,
        num_classes=10,
        training_strategy=args.training_strategy,
        beta=args.beta,
        threshold=args.threshold,
        bfunc=bfunc,
        in_ch=2,
        input_size=48,
    ).to(device)
else:
    net = VGGSNN(
        T=args.T_step,
        num_classes=10,
        training_strategy=args.training_strategy,
        beta=args.beta,
        threshold=args.threshold,
        spike_grad=spike_grad,
        in_ch=2,
        input_size=48,
    ).to(device)

# print(net)
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
        _ = model(x)
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
#     "conv_type": str(type(net.stage1[0].layer)),
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

    # if test_acc > test_best:
    #     test_best = test_acc
    #     torch.save(net.state_dict(), save_root + '/model.pth')

    # scheduler.step(loss_test)
    scheduler.step()
print(max(test_acc_hist))
#
# data={
#     "best acc": max(test_acc_hist),
#     "train loss": loss_hist,
#     "train acc": train_acc_hist,
#     "test acc": test_acc_hist,
# }
# with open(save_root+'/data.json', 'w') as json_file:
#     json.dump(data, json_file, indent=4)
