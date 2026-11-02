import os
import sys
import json
import math
import random
import argparse
from collections import OrderedDict

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from snntorch import surrogate

from VGGSNN import VGGSNN

from attack import (
    evaluate_clean,
    evaluate_transfer_matrix,
    save_results_json,
)


# ============================================================
# args
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument("--trial_num", type=int, default=1)
parser.add_argument("--batch", type=int, default=50)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--T_step", type=int, default=10)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--threshold", type=float, default=1.0)

parser.add_argument("--attack", type=str, default="all", choices=["fgsm", "rfgsm", "pgd", "all"])
parser.add_argument("--eps", type=float, default=8 / 255)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--no_random_start", action="store_true")
parser.add_argument("--no_reset_state", action="store_true")


parser.add_argument("--bfunc", type=str, default="gaussian")
parser.add_argument("--amp", type=float, default=2.5)
parser.add_argument("--phase", type=float, default=-1.0)

args = parser.parse_args()


# ============================================================
# basic setup
# ============================================================
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(args.trial_num * 100 + args.trial_num)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device.index)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0)

mean = (0.0, 0.0)
std = (1.0, 1.0)

reset_state = not args.no_reset_state


# ============================================================
# data: CIFAR10-DVS
# ============================================================
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)
CLASS_TO_LABEL = {c: i for i, c in enumerate(CLASS_NAMES)}


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
        data = obj["frames"]  # [T, 2, 128, 128]
        data = torch.from_numpy(data).float()

        if self.transform is not None:
            data = self.transform(data)

        return data, label


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


frames_root = "..."

transform_test = transforms.Compose([
    transforms.Resize(size=(48, 48)),
])

testset = CIFAR10DVSNPZDataset(
    root=frames_root,
    train=False,
    train_ratio=0.9,
    transform=transform_test,
    seed=0,
)

test_loader = DataLoader(
    testset,
    batch_size=args.batch,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
    pin_memory=True,
    worker_init_fn=seed_worker,
    persistent_workers=True if args.num_workers > 0 else False,
    prefetch_factor=4 if args.num_workers > 0 else None,
)


# ============================================================
# model wrapper
# ============================================================
class AttackModelWrapper(nn.Module):

    def __init__(self, model, reduce="mean"):
        super().__init__()
        self.model = model
        self.reduce = reduce

    def forward(self, x):
        return self.model.forward_logits(x, reduce=self.reduce)


# ============================================================
# model helpers
# ============================================================
def build_model(
    training_strategy,
    device,
    T_step,
    beta,
    threshold,
    spike_grad=None,
    bfunc=None,
):
    if training_strategy == "auSF":
        model = VGGSNN(
            T=T_step,
            num_classes=10,
            training_strategy="auSF",
            beta=beta,
            threshold=threshold,
            bfunc=bfunc,
            in_ch=2,
            input_size=48,
        ).to(device)
    else:
        model = VGGSNN(
            T=T_step,
            num_classes=10,
            training_strategy=training_strategy,
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            in_ch=2,
            input_size=48,
        ).to(device)

    return model


def get_model_path(training_strategy, args):
    base_root = "..."

    if training_strategy == "BP":
        save_root = (
            f"{base_root}/SCNN_{args.T_step}/trial_{args.trial_num}/BP/Atan"
        )
    elif training_strategy == "uSF":
        save_root = (
            f"{base_root}/SCNN_{args.T_step}/trial_{args.trial_num}/uSF/Atan"
        )
    elif training_strategy == "auSF":
        save_root = (
            f"{base_root}/SCNN_{args.T_step}/trial_{args.trial_num}/"
            f"auSF/{args.bfunc}/amp_{args.amp}/phase_{args.phase}"
        )
    else:
        raise ValueError(f"Unknown training_strategy: {training_strategy}")

    return os.path.join(save_root, "model.pth")


def load_all_models(args, device):
    spike_grad = surrogate.atan(alpha=2.0)

    if args.bfunc == "gaussian":
        bfunc = lambda t: torch.exp(-(args.amp * (t + args.phase) ** 2))
    else:
        raise ValueError(f"Unsupported bfunc: {args.bfunc}")

    models = OrderedDict()

    for strategy in ["BP", "uSF", "auSF"]:
        model = build_model(
            training_strategy=strategy,
            device=device,
            T_step=args.T_step,
            beta=args.beta,
            threshold=args.threshold,
            spike_grad=spike_grad,
            bfunc=bfunc,
        )

        model_path = get_model_path(strategy, args)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cannot find model file for {strategy}: {model_path}"
            )

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Very important:
        # Your training/testing uses reduce="mean".
        # Do not use the default reduce="sum" here.
        model = AttackModelWrapper(model, reduce="mean").to(device)
        model.eval()

        models[strategy] = model

    return models


# ============================================================
# attack config
# ============================================================
def get_attack_kwargs(attack_name, args):
    if attack_name == "fgsm":
        return {
            "eps": args.eps,
        }

    if attack_name == "rfgsm":
        return {
            "eps": args.eps,
            "alpha": args.alpha,
        }

    if attack_name == "pgd":
        return {
            "eps": args.eps,
            "alpha": args.alpha,
            "steps": args.steps,
            "random_start": not args.no_random_start,
        }

    raise ValueError(f"Unknown attack: {attack_name}")


# ============================================================
# main attack
# ============================================================
def run_one_attack(models, attack_name, args):
    attack_kwargs = get_attack_kwargs(attack_name, args)

    raw_results = evaluate_transfer_matrix(
        models=models,
        loader=test_loader,
        attack_name=attack_name,
        attack_kwargs=attack_kwargs,
        loss_fn=loss_fn,
        mean=mean,
        std=std,
        device=device,
        reset_state=reset_state,
    )

    save_root = "..."

    if attack_name == "pgd":
        attack_dir = os.path.join(save_root, f"{attack_name}_steps{args.steps}")
    else:
        attack_dir = os.path.join(save_root, attack_name)

    os.makedirs(attack_dir, exist_ok=True)

    save_name = os.path.join(attack_dir, f"trial_{args.trial_num}.json")

    save_results_json(
        save_path=save_name,
        clean_acc=clean_results,
        attack_name=attack_name,
        attack_kwargs=attack_kwargs,
        raw_results=raw_results,
        extra={
            "dataset": "CIFAR10-DVS",
            "trial_num": args.trial_num,
            "T_step": args.T_step,
            "beta": args.beta,
            "threshold": args.threshold,
            "reset_state": reset_state,
            "reduce": "mean",
            "eps": args.eps,
            "alpha": args.alpha,
            "steps": args.steps,
        },
    )


def main():
    global clean_results

    models = load_all_models(args, device)

    clean_results = OrderedDict()

    for name, model in models.items():
        acc = evaluate_clean(
            model=model,
            loader=test_loader,
            mean=mean,
            std=std,
            device=device,
            reset_state=reset_state,
        )
        clean_results[name] = acc
        #print(f"{name:>5} clean acc: {acc * 100:.2f}%")

    if args.attack == "all":
        for attack_name in ["fgsm", "rfgsm", "pgd"]:
            run_one_attack(models, attack_name, args)
    else:
        run_one_attack(models, args.attack, args)


if __name__ == "__main__":
    main()