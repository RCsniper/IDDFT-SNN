import os
import sys
import json
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import surrogate

from ResNet_19 import ResNet19
from attack import (
    evaluate_clean,
    evaluate_transfer_matrix,
    extract_metric_matrix,
    #print_matrix,
    save_results_json,
)


# ============================================================
# args
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument('--trial_num', type=int, default=1)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda:5')
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--T_step', type=int, default=4)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--threshold', type=float, default=1.0)

parser.add_argument('--attack', type=str, default='fgsm', choices=['fgsm', 'rfgsm', 'pgd', 'all'])
parser.add_argument('--eps', type=float, default=8/255)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--no_random_start', action='store_true')
parser.add_argument('--no_reset_state', action='store_true')

# auSF params
parser.add_argument('--bfunc', type=str, default='gaussian')
parser.add_argument('--amp', type=float, default=3.0)
parser.add_argument('--phase', type=float, default=-1.5)

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
mean = (0.5071, 0.4867, 0.4408)
std  = (0.2675, 0.2565, 0.2761)
reset_state = not args.no_reset_state


# ============================================================
# data
# ============================================================
log_root = "..."
data_path = os.path.join(log_root, 'data', 'cifar_100')

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

cifar100_test = datasets.CIFAR100(
    data_path, train=False, download=True, transform=test_transform
)

test_loader = DataLoader(
    cifar100_test,
    batch_size=args.batch,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=True if args.num_workers > 0 else False,
    prefetch_factor=4 if args.num_workers > 0 else None,
)


# ============================================================
# model helpers
# ============================================================
def build_model(training_strategy, device, T_step, beta, threshold,
                spike_grad=None, bfunc=None):
    if training_strategy == "auSF":
        model = ResNet19(
            num_classes=100,
            training_strategy="auSF",
            beta=beta,
            threshold=threshold,
            bfunc=bfunc,
            T=T_step,
        ).to(device)
    else:
        model = ResNet19(
            T=T_step,
            num_classes=100,
            training_strategy=training_strategy,
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
        ).to(device)
    return model



def get_model_path(training_strategy, args):
    if training_strategy == 'BP':
        save_root = f"..."
    elif training_strategy == 'uSF':
        save_root = f"..."
    elif training_strategy == 'auSF':
        save_root = f"..."
    else:
        raise ValueError(f"Unknown training_strategy: {training_strategy}")

    return os.path.join(save_root, "model.pth")



def load_all_models(args, device):
    spike_grad = surrogate.atan(alpha=2.0)
    bfunc = lambda t: torch.exp(-(args.amp * (t + args.phase) ** 2))

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
            raise FileNotFoundError(f"Cannot find model file for {strategy}: {model_path}")

        #print(f"Loading {strategy} model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models[strategy] = model

    return models


# ============================================================
# attack config
# ============================================================
def get_attack_kwargs(attack_name, args):
    if attack_name == 'fgsm':
        return {
            'eps': args.eps,
        }
    if attack_name == 'rfgsm':
        return {
            'eps': args.eps,
            'alpha': args.alpha,
        }
    if attack_name == 'pgd':
        return {
            'eps': args.eps,
            'alpha': args.alpha,
            'steps': args.steps,
            'random_start': not args.no_random_start,
        }
    raise ValueError(f"Unknown attack: {attack_name}")


# ============================================================
# main
# ============================================================
def run_one_attack(models, attack_name, args):
    names = list(models.keys())
    attack_kwargs = get_attack_kwargs(attack_name, args)

    #print(f"\nEvaluating {attack_name.upper()} transfer matrix ...")
    #print(f"attack_kwargs = {json.dumps(attack_kwargs)}")

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

    # adv_acc_matrix = extract_metric_matrix(raw_results, 'adv_acc_full')
    # robust_acc_matrix = extract_metric_matrix(raw_results, 'robust_acc')

    #print_matrix(adv_acc_matrix, f"{attack_name.upper()} Adversarial Accuracy on Full Test Set")
    #print_matrix(robust_acc_matrix, f"{attack_name.upper()} Robust Accuracy on Clean-Correct Subset")

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
            'trial_num': args.trial_num,
            'T_step': args.T_step,
            'reset_state': reset_state,
            # 'model_names': names,
            # 'eps_255': args.eps * 255.0,
            # 'alpha_255': args.alpha * 255.0,
        },
    )
    #print(f"Saved results to: {save_name}")


def main():
    global clean_results

    models = load_all_models(args, device)

    #print("\nEvaluating clean accuracy ...")
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

    if args.attack == 'all':
        for attack_name in ['fgsm', 'rfgsm', 'pgd']:
            run_one_attack(models, attack_name, args)
    else:
        run_one_attack(models, args.attack, args)


if __name__ == "__main__":
    main()
